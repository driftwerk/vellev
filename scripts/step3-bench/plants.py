#!/usr/bin/env python3
"""Every plant of #169, each alone, each one filtered test: not for merging.
   Run from the repository's root. `--check` only asks that each pattern is
   there once. Each line of plants.jsonl says how a plant came out."""
import json, os, re, subprocess, sys
CHECK = '--check' in sys.argv
MB='crates/vellev-storage/src/merge_builds.rs'; ST='crates/vellev-storage/src/state.rs'
PA='crates/vellev-storage/src/partition.rs'; TI='crates/vellev-storage/src/timings.rs'
IX='crates/vellev-index/src/pool.rs'; CL='crates/vellev-index/src/clustered.rs'; VE='crates/vellev-index/src/vector.rs'
NO='crates/vellev-node/src/lib.rs'; ME='crates/vellev/src/metrics.rs'; BE='crates/vellev/tests/step3_bench.rs'
def lib(*f): return ['-p','vellev-storage','--lib','--locked']+list(f)+['--','--test-threads=2']
def sto(b,*f): return ['-p','vellev-storage','--test',b,'--locked','--','--test-threads=2']+list(f)
def idx(b,*f): return ['-p','vellev-index','--test',b,'--locked','--','--test-threads=2']+list(f)
def nod(b,*f): return ['-p','vellev-node','--test',b,'--locked','--','--test-threads=2']+list(f)
def ben(*f): return ['-p','vellev','--test','step3_bench','--locked','--','--test-threads=2']+list(f)
def met(*f): return ['-p','vellev','--test','metrics','--locked','--','--test-threads=2']+list(f)
SEC=sto('merge_pool_sections','a_clustered')
ALWAYS=sto('timings','rows_due_while_a_merge')
P=[
# storage: timings
('storage 1: the merge checkpoint times nothing',[(PA,'''        self.checkpoint(turn, &key, frame, position, Some(taking), |state| {
            let replaced = state.compacted(&merge, merged.into_inner());''','''        drop(taking);
        self.checkpoint(turn, &key, frame, position, None, |state| {
            let replaced = state.compacted(&merge, merged.into_inner());''')],ALWAYS),
('storage 2: rows due mid-flush re-stamped at the commit',[(TI,'let waited = due.map(|due| now.saturating_duration_since(due));','let waited = due.map(|_| Duration::ZERO);')],sto('timings','rows_due_while_a_flush')),
('storage 3: a spell restarted at the flush start',[(TI,'''        let due = state.due.take();
        state.taking = Some((taken, due));''','''        let due = state.due.take();
        state.capped = state.capped.map(|_| Instant::now());
        state.taking = Some((taken, due));''')],sto('timings','a_spell_at_the_backlog')),
('storage 4: clean recorded as flush',[(PA,'let _building = self.building(Holder::Clean).await;','let _building = self.building(Holder::Flush).await;')],sto('timings','each_holder')),
('storage 5: the plan held through the build',[(PA,'        drop(planning);\n        let MergePlanned {','        let MergePlanned {'),(PA,'''        let Some((built, shape)) = built.await? else {
            return Ok(self.given_up(planned));''','''        let built = built.await?;
        drop(planning);
        let Some((built, shape)) = built else {
            return Ok(self.given_up(planned));''')],sto('timings','a_merges_gather')),
('storage 6: stale counted as committed',[(PA,'            true => MergeEnd::Stale,\n','            true => MergeEnd::Committed,\n')],sto('merge_gather','counted_stale')),
('storage: a stale merge before its build counted committed',[(PA,'''        if self.stale(planned) {
            return Ok(Ended::Dropped(MergeEnd::Stale));
        }
        let dead = run.dead();''','''        if self.stale(planned) {
            return Ok(Ended::Dropped(MergeEnd::Committed));
        }
        let dead = run.dead();''')],sto('timings','each_merge')),
('storage: an empty flush timed as a hold',[(PA,'            building.built_nothing();\n','')],sto('timings','nothing_to_flush')),
('storage: the partition named with no bench keeping',[(TI,'''        if !self.keeping.load(Ordering::Acquire) {
            return;
        }''','''        let named = partition.to_string();
        if named.is_empty() || !self.keeping.load(Ordering::Acquire) {
            return;
        }''')],sto('timings_alloc')),
('storage: a separate count, half observed',[(TI,'    micros: AtomicU64,\n    /// Where an observe','    micros: AtomicU64,\n    count: AtomicU64,\n    /// Where an observe'),(TI,'        self.paused_here();\n        let micros','        self.paused_here();\n        self.count.fetch_add(1, Ordering::Relaxed);\n        let micros'),(TI,'            count: each.iter().sum(),','            count: self.count.load(Ordering::Relaxed),')],lib('a_histogram_read_half')),
('storage: rows noted after the lock',[(PA,'''                    partition.applying(|state| {
                        let made = state.apply_prepared(&applied, &log, prepared);
                        if made.is_ok() {
                            lag.applied(state.unflushed());
                        }
                        made
                    })
''','''                    let (made, unflushed) = partition.applying(|state| {
                        let made = state.apply_prepared(&applied, &log, prepared);
                        (made, state.unflushed())
                    });
                    if made.is_ok() {
                        lag.applied(unflushed);
                    }
                    made
''')],lib('rows_applied')),
('storage: open rows due not told',[(TI,'flush_due: taken.into_iter().chain(state.due).collect(),','flush_due: taken.into_iter().collect(),')],sto('timings','still_open')),
('storage: an open spell not told',[(TI,'            backlog: state.capped,\n','            backlog: None,\n')],sto('timings','still_open')),
('storage: freeing not counted for the node',[(PA,'                adopted.count_in(self.timings.freeing_counted());\n','')],lib('a_replaced_segment_is_counted')),
# storage: ends
('storage: a restore counted as changed',[(PA,'            true => MergeEnd::Stale,\n            false => MergeEnd::Stopped,','            true => MergeEnd::Changed,\n            false => MergeEnd::Stopped,')],sto('merge_gather','counted_stale')),
('storage: a restore under way not asked at a stop',[(PA,'        let restoring = self.restoring.load(Ordering::SeqCst);\n        Ended::Dropped(match','        let restoring = false;\n        Ended::Dropped(match')],sto('merge_gather','restore_under_way')),
('storage: stale not asked at the stop once built',[(PA,'''        if stopping.stopped(MergeStage::Encoding) {
            return Ok(self.given_up(planned));
        }
        let _keeping''','''        if stopping.stopped(MergeStage::Encoding) {
            return Ok(Ended::Dropped(MergeEnd::Stopped));
        }
        let _keeping''')],sto('merge_gather','once_built')),
('storage: a late stop hides the end',[(PA,'            Ok(Ended::Dropped(end)) => *end,\n            Err(_) => MergeEnd::Failed,','            Ok(Ended::Dropped(_)) | Err(_) if stopping.is_stopped() => MergeEnd::Stopped,\n            Ok(Ended::Dropped(end)) => *end,\n            Err(_) => MergeEnd::Failed,')],sto('timings','late_stop')),
('storage: given-up work timed as through',[(PA,'            false => Timed::MergeGivenUp(phase),','            false => Timed::Merge(phase),')],sto('timings','given_up')),
('storage: a gather given up not timed',[(PA,'                    let phase = Timed::MergeGivenUp(MergePhase::Gather);\n                    timings.record(phase, &key, gathering.elapsed());','                    let _ = gathering;')],sto('timings','given_up')),
('storage: a build given up timed as finished',[(PA,'Ok(None) | Err(_) => Timed::MergeGivenUp(MergePhase::Build),','Ok(None) | Err(_) => Timed::Merge(MergePhase::Build),')],sto('timings','given_up')),
('storage: gather timed as nothing',[(PA,'timings.record(Timed::Merge(MergePhase::Gather), &key, gathering.elapsed());','timings.record(Timed::Merge(MergePhase::Gather), &key, gathering.elapsed() * 0);')],sto('merge_gather','timed_as_the_copy')),
('storage: build timer from before the gather',[(PA,'''            let built = match rows(&stopping, build.as_deref(), on) {
                Some((file, vectors, with)) => {
                    let building = Instant::now();
''','''            let building = Instant::now();
            let built = match rows(&stopping, build.as_deref(), on) {
                Some((file, vectors, with)) => {
''')],sto('merge_gather','timed_as_the_copy')),
('storage: the plan held through the gather',[(PA,'        drop(planning);\n        let MergePlanned {','        let MergePlanned {'),(PA,'''        let Some((built, shape)) = built.await? else {
            return Ok(self.given_up(planned));''','''        let built = built.await?;
        drop(planning);
        let Some((built, shape)) = built else {
            return Ok(self.given_up(planned));''')],sto('merge_gather','timed_as_the_copy')),
# slots
('slots: every build on slot 0',[(MB,'        let slot = lock(&self.slots_free).pop();\n        Some(slot.expect("a slot goes back before the turn that held it"))','        Some(0)')],lib('nesting')),
('slots: a slot never given back',[(MB,'            lock(&self.builds.slots_free).push(slot);\n','            let _ = slot;\n')],lib('nesting')),
('slots: a slot goes back after its turn',[(MB,'pub(crate) struct MergeBuild {\n    counted: Counted,\n    _turn: OwnedSemaphorePermit,\n}','pub(crate) struct MergeBuild {\n    _turn: OwnedSemaphorePermit,\n    counted: Counted,\n}')],lib('a_slot_goes_back')),
('slots: sections off the slot pool',[(MB,'        let slot = self.counted.slot?;\n        Some(&*self.counted.builds.pools[slot])','        let _ = self.counted.slot?;\n        None')],SEC),
('slots: sections always on pool 0',[(MB,'Some(&*self.counted.builds.pools[slot])','{ let _ = slot; Some(&*self.counted.builds.pools[0]) }')],sto('merge_pool_sections','two_merges')),
('slots: the gather off the pool',[(PA,'            on.section("gathering", || {','            (|| {'),(PA,'                Some((file, Some(vectors), shape))\n            })\n','                Some((file, Some(vectors), shape))\n            })()\n')],SEC),
('slots: the clustered build off the pool',[(ST,'ClusteredIndex::build_stopping_on(schema.distance, vectors, stop, on)','ClusteredIndex::build_stopping_on(schema.distance, vectors, stop, vellev_index::pool::CallersPool)')],SEC),
('slots: parts off the pool',[(PA,'on.section("parts", || state::Parts::of(&file, &indexes))','state::Parts::of(&file, &indexes)')],SEC),
('slots: encoding off the pool',[(PA,'on.section("encoding", || encode_as(Kind::Segment, format, &file))?','encode_as(Kind::Segment, format, &file)?')],SEC),
('slots: summarising off the pool',[(ST,'on.section("summarising", || index.rows.summarise());','index.rows.summarise();')],SEC),
('slots: finishing off the pool',[(PA,'on.section("finishing", || gathered.finish(format))','gathered.finish(format)')],SEC),
('slots: top-level CPU not counted',[(PA,'build.builds().spent_on_top(after.saturating_sub(before));','let _ = (build, before, after);')],SEC),
# index
('index: clustering pass off the given pool',[(CL,'let next: Vec<usize> = on.pass("clustering", || {','let next: Vec<usize> = CallersPool.pass("clustering", || {')],idx('merge_pool','a_build_given')),
('index: clustered coding off the given pool',[(CL,'on.pass("coding", || {','CallersPool.pass("coding", || {')],idx('merge_pool','a_build_given')),
('index: binary coding off the given pool',[(VE,'let factors = on.pass("coding", || {','let factors = CallersPool.pass("coding", || {')],idx('merge_pool','a_build_given')),
('index: slots not shares of the threads',[(IX,'.num_threads(threads / slots)','.num_threads(threads)')],idx('merge_pool','merge_slots')),
('index: merge pool threads never lowered',[(IX,'.start_handler(move |_| lowered_by(nice))\n                .build()\n                .expect("a merge slot\'s threads")','.build()\n                .expect("a merge slot\'s threads")')],idx('merge_pool_nice')),
# node
('node: threads under the builds let through',[(NO,'        if threads < builds {','        if false && threads < builds {')],nod('node','a_merge_pool_that')),
('node: nice past 19 let through',[(NO,'            0..=19 => None,','            0.. => None,')],nod('node','a_merge_pool_that')),
('node 8: the let-go flush timed as ordinary',[(NO,'timings.record(Timed::LetGoFlushWait, partition, times.waited);','timings.record(Timed::FlushDue, partition, times.waited);')],nod('sweeps','the_flush_before_letting')),
# metrics
('metrics: merged segments counted as unmerged',[(ST,'        if !crate::blobs::merged_segment(&meta.key) {\n            self.unmerged += 1;','        if true || !crate::blobs::merged_segment(&meta.key) {\n            self.unmerged += 1;')],met('background_work')),
('metrics: the +Inf bucket left out',[(ME,'        each.push((f64::MAX, count - below));\n','')],met('background_work')),
('metrics: a turn given up served',[(ME,'MergePhase::GIVEN_UP.map(|phase|','MergePhase::ALL.map(|phase|')],met('background_work')),
# bench: the arms and marks
('bench: every arm asked of a pool: threads under the builds',[(NO,'        if threads < builds {','        if false && threads < builds {')],ben('every_arm')),
('bench: schedule: a turn in one place\'s batches',[(BE,'a_turn: whole(bench.rotate.as_secs() * rate, batch * active, "a turn")','a_turn: whole(bench.rotate.as_secs() * rate, batch, "a turn")')],ben('the_active_set')),
('bench: flush_rows not pinned',[(BE,'            && self.flush_rows == 10_000\n','')],ben('a_run_off')),
('bench: budget share not pinned',[(BE,'            && self.budget_share == 0.5\n','')],ben('a_run_off')),
('bench: active not pinned',[(BE,'            && self.active == 3\n','')],ben('a_run_off')),
('bench: rotation not pinned',[(BE,'            && self.rotate == Duration::from_secs(60)\n','')],ben('a_run_off')),
('bench: batch not pinned',[(BE,'            && self.batch == 100\n','')],ben('a_run_off')),
('bench: p99 of answered reads only',[(BE,'let all = latencies(&|_| true);','let all = latencies(&|read| read.answered == Answered::Yes);')],ben('reads_refused')),
('bench: a failed read allowed',[(BE,'(self.failed_reads > 0).then(','(false && self.failed_reads > 0).then(')],ben('an_arm_with_a_failed')),
('bench: under its rate allowed',[(BE,'            wrote_too_little.then(|| {','            (false && wrote_too_little).then(|| {')],ben('an_arm_that_wrote_under')),
('bench: refused share not checked',[(BE,'(self.refused_reads as f64 > REFUSED_AT_MOST * self.reads as f64)','(false && self.refused_reads as f64 > REFUSED_AT_MOST * self.reads as f64)')],ben('an_arm_that_refused')),
('bench: no reads allowed',[(BE,'(self.reads == 0).then(|| "no reads".into()),','(false && self.reads == 0).then(|| "no reads".into()),')],ben('a_run_short')),
('bench: a reader lost allowed',[(BE,'(self.readers_lost > 0).then(','(false && self.readers_lost > 0).then(')],ben('a_run_short')),
('bench: late gate off',[(BE,'(self.late_most > LATE_AT_MOST.as_secs_f64())','(false && self.late_most > LATE_AT_MOST.as_secs_f64())')],ben('a_run_short')),
('bench: the late window gone',[(BE,'    now >= soak_end + LATE_AT_MOST\n','    now >= soak_end\n')],ben('a_batch_late')),
('bench: debt looks ahead',[(BE,'points.next_if(|point| point.0 <= t)','points.next_if(|point| point.0 <= t + 5.0)')],ben('merge_debt')),
('bench: the tail is the last instant',[(BE,'(!tail.is_empty()).then(|| tail.iter().sum::<f64>() / tail.len() as f64)','tail.last().copied()')],ben('debt_is_the_tails')),
('bench: the tail is 20 minutes',[(BE,'    let cycle = rotate.as_secs_f64() * PRELOADS.len() as f64;\n','    let cycle = rotate.as_secs_f64() * 20.0;\n')],ben('a_rising_debt')),
('bench: the report\'s tail is 20 minutes',[(BE,'tails.push(tail_mean(&series, last_cycle).unwrap_or(last));','tails.push(tail_mean(&series, last_from).unwrap_or(last));')],ben('the_tail_reported')),
('bench: flush_due: whole wait against the median',[(BE,'due < 2.0 * held && due - held < build','due < 2.0 * held && due < build')],ben('flush_due')),
('bench: flush_due: excess clause dropped',[(BE,'due < 2.0 * held && due - held < build','{ let _ = build; due < 2.0 * held }')],ben('flush_due')),
('bench: flush_due: own-time clause dropped',[(BE,'due < 2.0 * held && due - held < build','{ let _ = held; due - held < build }')],ben('flush_due')),
('bench: all marks without flush_due',[(BE,'        && passed("flush_due")\n','')],ben('beats_b')),
('bench: margin not held to the spread',[(BE,'Some(spread) if (b.read_p99_ms - arm.read_p99_ms) / b.read_p99_ms <= spread => {','Some(spread) if false && (b.read_p99_ms - arm.read_p99_ms) / b.read_p99_ms <= spread => {')],ben('beats_b')),
('bench: no spread still a win',[(BE,'            None => Beats::Undecided,\n            Some(spread)','            None => Beats::Yes,\n            Some(spread)')],ben('beats_b')),
('bench: (b) named having failed',[(BE,'            Some(false) => "none met every mark",\n','            Some(false) => "b",\n')],ben('beats_b')),
('bench: (b) must pass',[(BE,'    if !arm.valid || !b.valid {\n        return Beats::Undecided;','    if !arm.valid || !b.valid || b.all_marks != Some(true) {\n        return Beats::Undecided;')],ben('beats_b')),
('bench: settings not compared',[(BE,'if theirs != config {','if false && theirs != config {')],ben('arms_are_compared')),
('bench: SHA not compared',[(BE,'if other.node["sha"] != run.node["sha"] {','if false && other.node["sha"] != run.node["sha"] {')],ben('arms_are_compared')),
('bench: rows by tenant not compared',[(BE,'if written != &load["writes"]["scheduled_by_tenant"] || written.is_null() {','if false && (written != &load["writes"]["scheduled_by_tenant"] || written.is_null()) {')],ben('arms_are_compared')),
('bench: an unstamped SHA allowed',[(BE,'if sha.is_empty() {','if false && sha.is_empty() {')],ben('arms_are_compared')),
('bench: a dirty tree allowed',[(BE,'if sha.ends_with("-dirty") {','if false && sha.ends_with("-dirty") {')],ben('arms_are_compared')),
('bench: another machine allowed',[(BE,'if other.node["machine"] != run.node["machine"] {','if false && other.node["machine"] != run.node["machine"] {')],ben('arms_are_compared')),
('bench: preload not checked',[(BE,'if held != expected {','if false && held != expected {')],ben('arms_are_compared')),
('bench: an invalid (a) still the bound',[(BE,'        .filter(|_| valid)\n','')],ben('an_invalid_arm')),
('bench: noise from invalid runs',[(BE,'(Some(b), Some(b2)) if b.valid && b2.valid => json!({','(Some(b), Some(b2)) => json!({')],ben('the_noise_floor')),
('bench: a stage past the last clamped',[(BE,'"+Inf" => json!(format!("over {} s", last_bound.clone().unwrap_or_default())),','"+Inf" => json!(last_bound.clone().and_then(|b| b.parse::<f64>().ok())),')],ben('a_stages_p99')),
('bench: runtime threads unnamed in pools',[(BE,'        (RUNTIME_THREADS, "tokio"),\n','        ("tokio-runtime-w", "tokio"),\n')],ben('a_threads_pool')),
# bench: steal
('bench: steal: the average gate off',[(BE,'(Some(steal), Some(_)) if steal > STEAL_AT_MOST => Some(format!(','(Some(steal), Some(_)) if false && steal > STEAL_AT_MOST => Some(format!(')],ben('a_steady_theft')),
('bench: steal: the time gate off',[(BE,'(Some(_), Some(stolen_for)) if stolen_for > STOLEN_FOR_AT_MOST => Some(format!(','(Some(_), Some(stolen_for)) if false && stolen_for > STOLEN_FOR_AT_MOST => Some(format!(')],ben('an_arm_its_host')),
('bench: steal: time by a higher mark',[(BE,'if ticks > 0 && stolen as f64 / ticks as f64 > STEAL_AT_MOST {','if ticks > 0 && stolen as f64 / ticks as f64 > 0.5 {')],ben('a_spell_of_steal_counts')),
('bench: steal: last share not the peak',[(BE,'*peak = peak.max(share);','*peak = share;')],ben('a_spell_of_steal_shows')),
('bench: steal: throttling read from the wrong line',[(BE,'.find_map(|line| line.strip_prefix("throttled_usec "))?','.find_map(|line| line.strip_prefix("nr_throttled "))?')],ben('a_cgroups')),
('bench: steal: accounted whatever the kernel shows',[(BE,'(!stolen.is_empty()).then(|| stolen.iter().any(|ticks| *ticks > 0))','(!stolen.is_empty()).then_some(true)')],ben('a_kernel_that_shows')),
('bench: steal: unmeasured not said',[(BE,'        _ => format!("{said} (steal unmeasured)"),','        _ => said,')],ben('a_run_where_steal')),
# bench: windows, wiring
('bench: waits kept by when they ended',[(BE,'began.is_some_and(|began| began >= started && began <= ended)','{ let _ = began; event.at >= started && event.at <= ended }')],ben('a_wait_counts')),
('bench: open waits mix kinds',[(BE,'let kind = format!("{:?}", event.timed);','let kind = "all".to_string();')],ben('a_wait_counts')),
('bench: figures not of the soak',[(BE,'.map(|(key, n)| (*key, n - before.get(key).copied().unwrap_or(0)))','.map(|(key, n)| { let _ = before; (*key, *n) })')],ben('figures_are_of_the_soak')),
('wiring: merges ended read after settling',[(BE,'"merges_ended": since(&before.ended, &at_end.ended),','"merges_ended": since(&before.ended, &settled.ended),')],ben('figures_are_those_read')),
('wiring: let-go read after settling',[(BE,'"budget": at_end.let_go_budget - before.let_go_budget }','"budget": settled.let_go_budget - before.let_go_budget }')],ben('figures_are_those_read')),
('wiring: backlog read after settling',[(BE,'"refused": at_end.backlogged - before.backlogged,','"refused": settled.backlogged - before.backlogged,')],ben('figures_are_those_read')),
('wiring: largest read after settling',[(BE,'    let largest_at_soak_end = at_end.largest;\n','    let largest_at_soak_end = settled.largest;\n')],ben('figures_are_those_read')),
('wiring: top-level CPU read after settling',[(BE,'at_end.top_level_cpu.saturating_sub(before.top_level_cpu)','settled.top_level_cpu.saturating_sub(before.top_level_cpu)')],ben('figures_are_those_read')),
('wiring: events filtered by when they ended',[(BE,'.filter(|event| began_within(event, started, soak_end))','.filter(|event| event.at >= started && event.at <= soak_end)')],ben('a_wait_begun_in_the_soak')),
('wiring: a wait open at shutdown not in the mark',[(BE,'    flush_due.extend(&open_at_shutdown.0);\n','')],ben('a_wait_open_at_shutdown')),
('wiring: open waits of any time',[(BE,'let within = |began: &&Instant| **began >= started && **began <= ended;','let within = |began: &&Instant| { let _ = (started, ended); let _ = began; true };')],ben('a_wait_open_at_shutdown')),
('wiring: steal without the two ends',[(BE,'''    let steal_samples = with_ends(
        &sampled.steal,
        before.ticks,
        at_end.ticks,
        soaked.as_secs_f64(),
    );
''','''    let steal_samples = sampled.steal.clone();
''')],ben('steal_is_read_from')),
('wiring: a late read not voiding',[(BE,'read_as_it_ended: read_as_it_ended(at_end.taken_at, soak_end),','read_as_it_ended: true,')],ben('a_soak_end_read_late')),
('wiring: the rate of all rows',[(BE,'rows_written_a_second: writes.rows_in_soak as f64 / soaked,','rows_written_a_second: writes.rows as f64 / soaked,')],ben('the_rate_is_of_the_rows')),
# bench: slots and idle
('bench: idle share never large',[(BE,'"large": all_s > 0.0 && idle_s / all_s > IDLE_SHARE_LARGE,','"large": false,')],ben('a_merge_build_is_idle')),
('bench: idle by when it ended',[(BE,'let began = Duration::from_secs_f64((ended - took).max(0.0));','let began = Duration::from_secs_f64(*ended);')],ben('a_merge_build_is_idle')),
('bench: given-up work left out',[(BE,'            | Timed::MergeGivenUp(MergePhase::Gather | MergePhase::Build)\n','')],ben('work_given_up_counts')),
('bench: gathers left out',[(BE,'Timed::Merge(MergePhase::Gather | MergePhase::Build)','Timed::Merge(MergePhase::Build)')],ben('work_given_up_counts')),
('bench: lone time miscounted',[(BE,'            if running == 1 {\n                lone_s += at - last;\n            }\n','            if running == 2 {\n                lone_s += at - last;\n            }\n')],ben('slot_use')),
('bench: idle share ignores the slots',[(BE,'.map(|slots| 1.0 - slot_s / (busy_s * slots as f64)),','.map(|_| 1.0 - slot_s / busy_s),')],ben('slot_use')),
('bench: idle share of an arm with no slots',[(BE,'''        "idle_share": slots
            .filter(|_| busy_s > 0.0)''','''        "idle_share": Some(slots.unwrap_or(2))
            .filter(|_| busy_s > 0.0)''')],ben('slot_use')),
('bench: any build called lone',[(BE,'.filter(|one| spans.iter().filter(|other| overlaps(one, other)).count() == 1)','.filter(|one| spans.iter().filter(|other| overlaps(one, other)).count() >= 1)')],ben('slot_use')),
('bench: spans not clipped',[(BE,'let clipped = ended.min(soak_s);','let clipped = *ended;')],ben('slot_use')),
('bench: a cut-off build called lone',[(BE,'.filter(|one| one.0 && one.1 < soak_s)','.filter(|one| one.0)')],ben('slot_use')),
]
bad=0; out=[]
env=dict(os.environ,RAYON_NUM_THREADS='2')
for name,edits,args in P:
    texts={}
    for f,a,b in edits:
        s=texts.get(f) or open(f).read()
        if s.count(a)!=1:
            print(f'PATTERN {s.count(a)}: {name}: {a[:60]!r}',flush=True); bad+=1; s=None; break
        texts[f]=s.replace(a,b)
    if CHECK or s is None: continue
    for f,t in texts.items(): open(f,'w').write(t)
    try:
        r=subprocess.run(['cargo','test']+args,env=env,capture_output=True,text=True,timeout=900)
        o=r.stdout+r.stderr
    except subprocess.TimeoutExpired:
        o=''
        rec=dict(plant=name,state='TIMED OUT',failed=[],ran=0,error=[])
        print(json.dumps(rec),flush=True); out.append(rec)
        with open('plants.jsonl','a') as f: f.write(json.dumps(rec)+chr(10))
        subprocess.run(['git','checkout','--']+list(texts),capture_output=True)
        continue
    failed=re.findall(r'^test (\S+) \.\.\. FAILED',o,re.M)
    ran=len(re.findall(r'^test \S+ \.\.\. ',o,re.M))
    err=[l for l in o.splitlines() if l.startswith('error[') or l.startswith('error: could not compile')]
    state='NOT COMPILED' if err else ('FAILED' if failed else ('SURVIVED' if ran else 'NOTHING RAN'))
    rec=dict(plant=name,state=state,failed=failed,ran=ran,error=err[:2])
    print(json.dumps(rec),flush=True); out.append(rec)
    with open('plants.jsonl','a') as f: f.write(json.dumps(rec)+'\n')
    subprocess.run(['git','checkout','--']+list(texts),capture_output=True)
print(f'## {len(P)} plants, {bad} patterns missing',flush=True)
if not CHECK:
    print('## failed as planted:',sum(r['state']=='FAILED' for r in out),'of',len(out))
sys.exit(1 if bad else 0)
