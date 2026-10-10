//! Those who wait for a local model, and how a run of it is shared out
//! among them.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::oneshot;

use crate::{EmbedError, Progress, Purpose, Result};

/// A write of this many texts or fewer is a small one, and goes ahead of
/// its own tenant's larger writes.
pub(crate) const SMALL: usize = 4;
/// The most characters of a text a model is given. No model here reads
/// more than 512 tokens of a text, and a tokenizer keeps no word of more
/// than a hundred characters as a word: so what comes after these is
/// read by none, unless the text has long runs of space in it, which
/// count here and are nothing to a tokenizer.
pub(crate) const LONGEST_TEXT: usize = 512 * 100;
/// What a text counts for in a run at the least and at the most, in
/// bytes: a short text costs the model something, and it reads no more
/// of a long one than about this.
pub(crate) const LIGHTEST: usize = 64;
pub(crate) const HEAVIEST: usize = 2_048;
/// How many times over what waits ahead of a write is reckoned to take,
/// before the write is given up on for not having come to its turn: the
/// reckoning is by how long the last runs took, and the next may take
/// longer.
const SLACK: u32 = 4;

pub(crate) struct Job {
    pub(crate) texts: Vec<String>,
    pub(crate) purpose: Purpose,
    /// Whose it is: the model is shared out among those who ask of it,
    /// and not among what they ask.
    pub(crate) tenant: Arc<str>,
    pub(crate) reply: oneshot::Sender<Result<Vec<Vec<f32>>>>,
    /// What the caller is told through how long the model is given.
    pub(crate) progress: Option<Progress>,
}

/// A job and the vectors made for it so far.
pub(crate) struct Work {
    pub(crate) job: Job,
    pub(crate) done: Vec<Vec<f32>>,
    /// When it last got somewhere: when it came, when it was last given
    /// a vector, or when one ahead of it in its tenant's queue was done
    /// with. Nothing anyone else's work does is its getting somewhere.
    since: Instant,
}

impl Work {
    pub(crate) fn left(&self) -> usize {
        self.job.texts.len() - self.done.len()
    }

    /// The next `share` of the texts it has left.
    fn next(&self, share: usize) -> Vec<&str> {
        let texts = &self.job.texts[self.done.len()..][..share];
        texts.iter().map(String::as_str).collect()
    }

    /// Tell the caller how long it is reckoned to be before the model
    /// comes to this.
    pub(crate) fn allowed(&self, within: Duration) {
        if let Some(progress) = &self.job.progress {
            progress.allowed(within);
        }
    }

    fn refuse(self, refusal: EmbedError) {
        let _ = self.job.reply.send(Err(refusal));
    }
}

/// What a text counts for in a run: its length, but no less than a short
/// text and no more than the model reads of a long one. A run is of as
/// much as takes about as long as a run is to, and not of so many texts
/// whatever they are: so what one tenant sends cannot make a run long
/// for everyone.
pub(crate) fn weight(text: &str) -> usize {
    text.len().clamp(LIGHTEST, HEAVIEST)
}

/// `text`, no longer than [`LONGEST_TEXT`] characters: the rest is read
/// by no model, and is not carried to one. It is cut where a character
/// ends.
pub(crate) fn cut(mut text: String) -> String {
    if text.len() > LONGEST_TEXT {
        if let Some((at, _)) = text.char_indices().nth(LONGEST_TEXT) {
            text.truncate(at);
        }
    }
    text
}

/// One tenant's jobs of one kind, in the order they are to be made.
pub(crate) struct Queue {
    pub(crate) tenant: Arc<str>,
    pub(crate) works: VecDeque<Work>,
    /// The weight the tenant may send that it has not: its share of each
    /// run it waited through, less what it sent. A text goes when the
    /// tenant is owed its weight, so a tenant with long texts waits a
    /// few runs for each and one with short texts sends some in every
    /// run, and over a few runs each has had the same of the model's
    /// time. It is never more than a run's share and one text's weight,
    /// so that what a tenant could not send is not saved up to be had
    /// all at once later; and it is forgotten with the queue, when
    /// nothing is left in it.
    owed: usize,
}

/// How many texts of each job of each queue go into a run.
pub(crate) type Shares = Vec<Vec<usize>>;

/// The jobs of one kind that wait for the model, each tenant's in a
/// queue of its own, and the queues in the order their turns come.
///
/// A run is shared out among the tenants, a text at a time in turn, each
/// tenant sending while it is owed the weight of its next text: each run
/// owes each tenant that waits an equal share of the run's weight, which
/// a tenant keeps until it has sent it. So each tenant has as much of
/// the model's time as any other that wants it, however much each asks
/// for and however long its texts: one tenant's thousand texts are not
/// ahead of another's one, and one tenant's long texts do not have the
/// model for longer than another's short ones. What a tenant has of a
/// run goes to its own jobs in the order they came, but that its small
/// writes, of [`SMALL`] texts or fewer, go ahead of its larger.
#[derive(Default)]
pub(crate) struct Line {
    pub(crate) queues: VecDeque<Queue>,
    /// Which queue the next run begins with, as the queues stood when
    /// the last was shared out: the one after the queue its last text
    /// went to, so that the tenants take turns across runs as within
    /// one; or the first that was owed a text that did not fit in what
    /// was left of the run, which is thereby first into the next.
    begins_next: Option<usize>,
}

impl Line {
    pub(crate) fn is_empty(&self) -> bool {
        self.queues.is_empty()
    }

    pub(crate) fn works(&self) -> impl Iterator<Item = &Work> {
        self.queues.iter().flat_map(|queue| &queue.works)
    }

    pub(crate) fn push(&mut self, job: Job, now: Instant) {
        let at = self
            .queues
            .iter()
            .position(|queue| queue.tenant == job.tenant);
        let at = at.unwrap_or_else(|| {
            self.queues.push_back(Queue {
                tenant: job.tenant.clone(),
                works: VecDeque::new(),
                owed: 0,
            });
            self.queues.len() - 1
        });
        let works = &mut self.queues[at].works;
        // Small by what was asked, not by what is left: a large write
        // near its end does not come to go ahead of others.
        let small = |work: &Work| work.job.texts.len() <= SMALL;
        let work = Work {
            job,
            done: Vec::new(),
            since: now,
        };
        let behind = match small(&work) {
            true => works.iter().position(|work| !small(work)),
            false => None,
        };
        works.insert(behind.unwrap_or(works.len()), work);
    }

    /// Let go of the jobs whose callers have gone away. Those behind one
    /// in its tenant's queue have got somewhere by it.
    pub(crate) fn drop_gone(&mut self, now: Instant) {
        for queue in &mut self.queues {
            let mut left = false;
            queue.works.retain_mut(|work| {
                let gone = work.job.reply.is_closed();
                if left {
                    work.since = now;
                }
                left |= gone;
                !gone
            });
        }
        self.queues.retain(|queue| !queue.works.is_empty());
    }

    /// Everyone in the line has got somewhere, as when a model they all
    /// waited on to load has loaded.
    pub(crate) fn all_got_somewhere(&mut self, now: Instant) {
        let works = self.queues.iter_mut().flat_map(|queue| &mut queue.works);
        works.for_each(|work| work.since = now);
    }

    /// How many texts of each job go into a run of `most` texts and
    /// `heaviest` weight at the most, and of one text at the least. The
    /// tenants that are `barred` have none of it.
    ///
    /// A run of no stated weight, as when nothing is known of the model's
    /// pace, is shared out a text at a time among the tenants in turn,
    /// and no one is owed anything for it.
    ///
    /// Of a run of a stated weight, each tenant that waits is owed an
    /// equal share, over what it was owed before. A text at a time is
    /// given to each tenant in turn, in the order they stand, for as long
    /// as the tenant is owed its next text's weight and the text fits in
    /// what is left of the run. Then what is left of the run that no one
    /// is owed goes the same way to whoever has a text that fits it, so
    /// that the model is not kept from work that waits: a tenant's
    /// unused share passes to the next in turn. Each text goes to the
    /// first of its tenant's jobs that has one left.
    ///
    /// A tenant owed a text that did not fit in what was left is first
    /// into the next run, where it fits, the run being of one text at
    /// the least. So a tenant's first text is in a run within as many
    /// runs as it takes to be owed it, and one more for each other tenant
    /// that was owed a text there was no room for: with T tenants
    /// waiting, a text of weight w and runs of weight W, that is
    /// T * (ceil(w / W) + 1) runs at the most, and one run when the
    /// others leave room for it.
    pub(crate) fn shared_out(
        &mut self,
        most: usize,
        heaviest: usize,
        barred: impl Fn(&str) -> bool,
    ) -> Shares {
        let mut shares: Shares = self
            .queues
            .iter()
            .map(|queue| vec![0; queue.works.len()])
            .collect();
        let waiting: Vec<usize> = (0..self.queues.len())
            .filter(|at| !barred(&self.queues[*at].tenant))
            .collect();
        if waiting.is_empty() {
            return shares;
        }
        let stated = heaviest != usize::MAX;
        if stated {
            let share = (heaviest / waiting.len()).max(1);
            for at in &waiting {
                let owed = &mut self.queues[*at].owed;
                *owed = owed.saturating_add(share).min(share + HEAVIEST);
            }
        }
        let (mut texts, mut weighs) = (0usize, 0usize);
        let mut last_served = None;
        // The first that was owed a text that did not fit.
        let mut did_not_fit = None;
        // What each is owed first, and then what no one is.
        for owed_only in [true, false] {
            if owed_only && !stated {
                continue;
            }
            let mut given = true;
            while given && texts < most {
                given = false;
                for at in &waiting {
                    if texts >= most {
                        break;
                    }
                    let queue = &mut self.queues[*at];
                    let next = queue.works.iter().zip(shares[*at].iter_mut());
                    let next = { next }.find(|(work, share)| **share < work.left());
                    let Some((work, share)) = next else {
                        continue;
                    };
                    let weight = weight(&work.job.texts[work.done.len() + *share]);
                    let is_owed = queue.owed >= weight;
                    let fits = texts == 0 || weighs.saturating_add(weight) <= heaviest;
                    if owed_only && is_owed && !fits && did_not_fit.is_none() {
                        did_not_fit = Some(*at);
                    }
                    if !fits || (owed_only && !is_owed) {
                        continue;
                    }
                    if stated {
                        queue.owed = queue.owed.saturating_sub(weight);
                    }
                    weighs = weighs.saturating_add(weight);
                    *share += 1;
                    texts += 1;
                    given = true;
                    last_served = Some(*at);
                }
            }
        }
        let after_last = last_served.map(|last| last + 1);
        self.begins_next = did_not_fit.or(after_last);
        shares
    }

    /// The tenants any job of whose has some of `shares`.
    pub(crate) fn tenants_sharing(&self, shares: &Shares) -> Vec<Arc<str>> {
        self.queues
            .iter()
            .zip(shares)
            .filter(|(_, shares)| shares.iter().any(|share| *share > 0))
            .map(|(queue, _)| queue.tenant.clone())
            .collect()
    }

    /// The texts of each job that has any of `shares`, in the order the
    /// jobs stand.
    pub(crate) fn texts<'a>(&'a self, shares: &Shares) -> Vec<Vec<&'a str>> {
        let queues = self.queues.iter().zip(shares);
        let works = queues.flat_map(|(queue, shares)| queue.works.iter().zip(shares));
        works
            .filter(|(_, share)| **share > 0)
            .map(|(work, share)| work.next(*share))
            .collect()
    }

    /// Take what was `made` for each job that had any of `shares`, in
    /// the order the jobs stand, and answer the callers that are thereby
    /// done or failed. A job nothing was made for and nothing is said
    /// of stands as it was. The tenants keep their order, less those
    /// with nothing left, turned so that the one to begin the next run
    /// stands first.
    ///
    /// The tenants a job of whose was failed, each with how.
    pub(crate) fn settle(
        &mut self,
        shares: Shares,
        made: Vec<Option<Result<Vec<Vec<f32>>>>>,
        now: Instant,
    ) -> Vec<(Arc<str>, EmbedError)> {
        let mut made = made.into_iter();
        let mut failed = Vec::new();
        let mut kept: VecDeque<Queue> = VecDeque::new();
        // Where the next run begins, among the queues kept: with the one
        // that is to begin it, or the first after it if that one has
        // nothing left, or the first of all if none is after.
        let begins_next = self.begins_next.take();
        let mut begin_at = None;
        for (at, (mut queue, shares)) in self.queues.drain(..).zip(shares).enumerate() {
            let mut left = VecDeque::with_capacity(queue.works.len());
            // Whether one ahead in the queue was done with in this run.
            let mut done_with = false;
            for (mut work, share) in queue.works.drain(..).zip(shares) {
                if done_with {
                    work.since = now;
                }
                // Only one with a share has anything made for it.
                let made = match share {
                    0 => None,
                    _ => made.next().expect("one for each job with a share"),
                };
                match made {
                    None => left.push_back(work),
                    Some(Ok(vectors)) => {
                        work.done.extend(vectors);
                        work.since = now;
                        match work.left() {
                            0 => {
                                done_with = true;
                                let _ = work.job.reply.send(Ok(work.done));
                            }
                            _ => left.push_back(work),
                        }
                    }
                    Some(Err(error)) => {
                        done_with = true;
                        failed.push((queue.tenant.clone(), error.clone()));
                        work.refuse(error);
                    }
                }
            }
            queue.works = left;
            if !queue.works.is_empty() {
                if begin_at.is_none() && begins_next.is_some_and(|begins| at >= begins) {
                    begin_at = Some(kept.len());
                }
                kept.push_back(queue);
            }
        }
        kept.rotate_left(begin_at.unwrap_or_default());
        self.queues = kept;
        failed
    }

    /// How long it is reckoned to be before the model comes to each job:
    /// with `each` taken over a text, one tenant's turn in as many as
    /// there are tenants, and what stands ahead of the job in its own
    /// tenant's queue. Another tenant's work adds to it only by being
    /// one more to take turns with.
    fn waits(&self, each: Duration) -> impl Iterator<Item = (&Queue, usize, Duration)> {
        let tenants = u32::try_from(self.queues.len()).unwrap_or(u32::MAX);
        self.queues.iter().flat_map(move |queue| {
            let mut ahead = 0u32;
            queue.works.iter().enumerate().map(move |(at, work)| {
                let wait = each
                    .saturating_mul(tenants)
                    .saturating_mul(ahead.saturating_add(1));
                ahead = ahead.saturating_add(u32::try_from(work.left()).unwrap_or(u32::MAX));
                (queue, at, wait)
            })
        })
    }

    /// Refuse the jobs the model is reckoned not to come to within
    /// `at_most`: they are told so now, to be sent again when there is
    /// less to do, and not kept to be told later. Those are the jobs
    /// deepest in the queues of the tenants with most waiting.
    ///
    /// How many were refused.
    pub(crate) fn refuse_held_too_long(
        &mut self,
        model: &str,
        each: Duration,
        at_most: Duration,
    ) -> usize {
        let held: Vec<(Arc<str>, usize, Duration)> = self
            .waits(each)
            .filter(|(_, _, wait)| *wait > at_most)
            .map(|(queue, at, wait)| (queue.tenant.clone(), at, wait))
            .collect();
        // From the back of each queue, so that those before stay where
        // they were counted.
        for (tenant, at, wait) in held.iter().rev() {
            let queue = self.queues.iter_mut().find(|queue| &queue.tenant == tenant);
            if let Some(work) = queue.and_then(|queue| queue.works.remove(*at)) {
                work.refuse(EmbedError::failed(
                    model,
                    format!(
                        "the model has more waiting for it than it will come to soon: this \
                         would wait about {wait:.0?}, and is not kept longer than {at_most:?}"
                    ),
                ));
            }
        }
        self.queues.retain(|queue| !queue.works.is_empty());
        held.len()
    }

    /// Tell each caller how long it is reckoned to be before the model
    /// comes to its job: several times what stands ahead of it is
    /// reckoned to take, from when it last got somewhere. A tenant kept
    /// from the model until some time is given until then besides.
    pub(crate) fn tell(
        &self,
        now: Instant,
        each: Option<Duration>,
        kept_until: impl Fn(&str) -> Option<Instant>,
    ) {
        for (queue, at, wait) in self.waits(each.unwrap_or_default()) {
            let work = &queue.works[at];
            let kept = kept_until(&queue.tenant).map_or(Duration::ZERO, |until| {
                until.saturating_duration_since(work.since)
            });
            let given = wait.saturating_mul(SLACK).saturating_add(kept);
            let spent = now.saturating_duration_since(work.since);
            work.allowed(given.saturating_sub(spent));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Answer = oneshot::Receiver<Result<Vec<Vec<f32>>>>;

    fn job(tenant: &str, texts: usize, progress: Option<Progress>) -> (Job, Answer) {
        let (reply, answer) = oneshot::channel();
        let job = Job {
            texts: vec!["x".into(); texts],
            purpose: Purpose::Document,
            tenant: tenant.into(),
            reply,
            progress,
        };
        (job, answer)
    }

    /// When each of `tenant`'s jobs last got somewhere, in their order.
    fn since(line: &Line, tenant: &str) -> Vec<Instant> {
        let queue = line.queues.iter().find(|queue| &*queue.tenant == tenant);
        queue.map_or_else(Vec::new, |queue| {
            queue.works.iter().map(|work| work.since).collect()
        })
    }

    /// A job gets somewhere when a vector is made for it, when one ahead
    /// of it in its own tenant's queue is done with or leaves, and when
    /// the model they all waited on loads; and not by anything another
    /// tenant's jobs do.
    #[test]
    fn a_job_gets_somewhere_by_its_own_tenants_doings_and_the_models_loading_and_no_others() {
        let secs = Duration::from_secs;
        let mut line = Line::default();
        let came = Instant::now();
        // a1 is small, and goes first; a2 and a3 are not, and follow.
        let (a1, _a1) = job("a", 2, None);
        let (a2, a2_answer) = job("a", 5, None);
        let (a3, _a3) = job("a", 5, None);
        let (b1, _b1) = job("b", 1, None);
        for job in [a1, a2, a3, b1] {
            line.push(job, came);
        }
        // A run of a1's two texts: a1 is done, and a2 and a3 behind it
        // have got somewhere by it. b1 has not.
        let shares = line.shared_out(2, usize::MAX, |tenant| tenant == "b");
        assert_eq!(shares, [vec![2, 0, 0], vec![0]]);
        let done = came + secs(1);
        line.settle(shares, vec![Some(Ok(vec![vec![1.0]; 2]))], done);
        assert_eq!(since(&line, "a"), [done, done]);
        assert_eq!(since(&line, "b"), [came]);
        // The next run begins with b, after a. A run of b1's text and one
        // of a3's: each got somewhere by its own vector, and a2, ahead of
        // a3, did not.
        assert_eq!(line.queues[0].tenant.as_ref(), "b");
        let shares = vec![vec![1], vec![0, 1]];
        let made = vec![Some(Ok(vec![vec![1.0]])), Some(Ok(vec![vec![1.0]]))];
        let vectored = came + secs(2);
        line.settle(shares, made, vectored);
        assert_eq!(since(&line, "a"), [done, vectored]);
        assert!(since(&line, "b").is_empty(), "b1 was done");
        // a2's caller goes away: a3, behind it, has got somewhere.
        drop(a2_answer);
        let left = came + secs(3);
        line.drop_gone(left);
        assert_eq!(since(&line, "a"), [left]);
        // The model everyone waited on has loaded: everyone has.
        let (c1, _c1) = job("c", 1, None);
        line.push(c1, came);
        let loaded = came + secs(4);
        line.all_got_somewhere(loaded);
        assert_eq!(since(&line, "a"), [loaded]);
        assert_eq!(since(&line, "c"), [loaded]);
    }

    /// What a tenant is owed is never more than its share of a run and
    /// one text's weight, however long it could not send what it was
    /// owed: here because a run holds no more than thirty-two texts,
    /// and its short texts are far less than its share.
    #[test]
    fn what_a_tenant_could_not_send_is_not_saved_up_without_bound() {
        let mut line = Line::default();
        let now = Instant::now();
        let (short, _short) = job("short", 400, None);
        let (other, _other) = job("other", 400, None);
        line.push(short, now);
        line.push(other, now);
        let budget = 16 * HEAVIEST;
        for _ in 0..8 {
            let shares = line.shared_out(32, budget, |_| false);
            let made = line
                .texts(&shares)
                .iter()
                .map(|texts| Some(Ok(vec![vec![1.0]; texts.len()])))
                .collect();
            line.settle(shares, made, now);
            for queue in &line.queues {
                assert!(queue.owed <= budget / 2 + HEAVIEST, "{}", queue.owed);
            }
        }
        // Each sent sixteen texts a run, of the weight of a short text,
        // and is owed all it may be.
        let owed: Vec<usize> = line.queues.iter().map(|queue| queue.owed).collect();
        assert_eq!(owed, [budget / 2 + HEAVIEST - 16 * LIGHTEST; 2]);
    }

    /// Each caller is told several times what is reckoned to stand ahead
    /// of it, less what it has spent since it last got somewhere, and
    /// how long its tenant is kept from the model besides.
    #[test]
    fn a_caller_is_told_what_stands_ahead_of_it_and_how_long_its_tenant_is_kept_from_the_model() {
        let (ms, secs) = (Duration::from_millis, Duration::from_secs);
        let mut line = Line::default();
        let now = Instant::now();
        let (says_a, hears_a) = Progress::heard();
        let (says_b, hears_b) = Progress::heard();
        let (a, _a) = job("a", 3, Some(says_a));
        let (b, _b) = job("b", 1, Some(says_b));
        line.push(a, now);
        line.push(b, now);
        assert_eq!(SLACK, 4);
        line.tell(now, Some(ms(10)), |tenant| {
            (tenant == "b").then_some(now + secs(5))
        });
        // a: 10 ms a text, two tenants in turn, nothing of its own ahead:
        // 20 ms, four times over, and nothing spent.
        assert_eq!(hears_a.borrow().allowed, Some(ms(80)));
        // b: the same, and the five seconds it is kept from the model.
        assert_eq!(hears_b.borrow().allowed, Some(ms(80) + secs(5)));
        // Told again later: less what was spent meanwhile.
        line.tell(now + ms(30), Some(ms(10)), |_| None);
        assert_eq!(hears_a.borrow().allowed, Some(ms(50)));
        assert_eq!(hears_b.borrow().allowed, Some(ms(50)));
    }
}
