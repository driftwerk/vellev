use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::time::Instant;
use vellev_core::{Embedding, EmbeddingProvider, HttpApi};

use crate::embedder::Shared;
use crate::{
    Bedrock, Embedder, Http, Provider, Result, DOCUMENTS_WITHIN, FAULT_STANDS,
    LONGEST_QUERY_WITHIN, QUERY_WITHIN,
};

/// How many local models a node has loaded at once, unless it is set up
/// otherwise: each is the better part of a gigabyte while it runs, and a
/// node is not to be run out of memory by the collections made on it.
pub const LOCAL_MODELS: usize = 2;
/// The local model a node runs unless it is set up to run others. A
/// collection names the model it embeds with: this is the one a node
/// will run for it unless told of others.
pub const DEFAULT_LOCAL_MODEL: &str = "BAAI/bge-base-en-v1.5";
/// How long a local model goes unused before a node lets it go, unless it
/// is set up otherwise.
pub const LOCAL_MODEL_IDLE: Duration = Duration::from_secs(15 * 60);

/// The fewest characters of a key a node is told to send an endpoint.
pub const SHORTEST_KEY: usize = 16;

/// The least time a node gives a model to answer, or keeps one unused,
/// whatever it is told.
const SHORTEST: Duration = Duration::from_secs(1);

/// How many models served elsewhere a node keeps what it knows of at
/// once: for each, a client to reach it with, the vectors it made for
/// queries, and whether it can be used. Which endpoints a collection may
/// name is the node's to say, but what model it asks them for, at what
/// size, and in which of Amazon's regions, is the collection's: so there
/// is no end to those a node can be asked for, and there is an end to
/// those it keeps.
const ELSEWHERE: usize = 64;

/// What embedding needs from the node rather than from a collection.
#[derive(Clone)]
pub struct Settings {
    /// Where local models are kept once fetched.
    pub cache: PathBuf,
    /// The bearer token for each HTTP endpoint that wants one, by its URL
    /// as collections give it. No other endpoint is sent a token.
    pub keys: HashMap<String, String>,
    /// How long a model found to be one that cannot be used is taken to
    /// stay so, and how often at most that is logged. It is taken as no
    /// less than a second and no more than an hour, whatever is given.
    pub fault_stands: Duration,
    /// How long a model is given to answer a query. One that takes longer
    /// has failed in a way that may pass, whatever the model and however
    /// it is reached. No less than a second, and no more than
    /// [`LONGEST_QUERY_WITHIN`].
    pub query_within: Duration,
    /// How long a model may go without making a vector for anyone, while
    /// it is not loading, before a call for documents that waits on it
    /// is given up on: not how long the call may take, which is as long
    /// as the model is at work. No less than a second.
    pub documents_within: Duration,
    /// How many local models are loaded at once at most, and one at the
    /// least. A collection whose model there is no room for cannot be
    /// embedded for until one of those loaded has gone unused for
    /// `local_model_idle`.
    pub local_models: usize,
    /// How long a local model goes unused before it is let go, to be
    /// loaded again when next asked for. No less than a second.
    pub local_model_idle: Duration,
    /// The node's own local model. A collection names the model it embeds
    /// with, and this is always one it may name.
    /// One of the places a node has for loaded models is kept for it,
    /// and it is not let go once loaded, so no other model can keep
    /// collections from it. Without one, no model is so kept.
    pub local_model: Option<String>,
    /// The other local models collections may embed with, by name,
    /// whichever organisation's copy is named and however it is cased.
    pub local_models_allowed: Vec<String>,
    /// The regions of Amazon's a collection may embed through Bedrock
    /// in. A collection names its own, and one not named here is one the
    /// node will not call: with none, no collection embeds through
    /// Bedrock. The calls the node has to Bedrock at once are shared
    /// equally among these, which are five at the most: each then has
    /// two of them, one kept for queries.
    pub bedrock_regions: Vec<String>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            cache: PathBuf::new(),
            keys: HashMap::new(),
            fault_stands: FAULT_STANDS,
            query_within: QUERY_WITHIN,
            documents_within: DOCUMENTS_WITHIN,
            local_models: LOCAL_MODELS,
            local_model_idle: LOCAL_MODEL_IDLE,
            local_model: Some(DEFAULT_LOCAL_MODEL.to_string()),
            local_models_allowed: Vec::new(),
            bedrock_regions: Vec::new(),
        }
    }
}

impl Settings {
    /// Whether a node could embed as these say, as far as can be told
    /// before it does: or what is wrong with them, for whoever set the
    /// node up to put right before it starts. What is wrong is a local
    /// model named that the release does not run, by a slip as likely as
    /// not, more local models than the node could ever have loaded, or
    /// more regions of Bedrock's than it has calls to share among.
    /// A node started so regardless leaves out what it cannot do.
    pub fn checked(&self) -> std::result::Result<(), String> {
        // A key is taken out of what an endpoint says by its pieces:
        // a short one is in every other word, and would be taken out of
        // all of them, the status and what stands for it included.
        let mut endpoints: Vec<&String> = self.keys.keys().collect();
        endpoints.sort();
        for endpoint in endpoints {
            if let Some(wrong) = crate::http::too_short(endpoint, &self.keys[endpoint]) {
                return Err(wrong);
            }
        }
        let most = crate::bedrock::MOST_REGIONS;
        // Counted as the node will call them: each once, however often
        // and however it is named.
        let regions = self.regions();
        if regions.len() > most {
            return Err(format!(
                "a node embeds through Bedrock in {most} regions at the most, each with a \
                 share of its calls to itself, and not in the {} of {}",
                regions.len(),
                regions.join(", ")
            ));
        }
        if self.query_within > LONGEST_QUERY_WITHIN {
            return Err(format!(
                "a search is kept waiting for its text's vector for {LONGEST_QUERY_WITHIN:?} at \
                 the most, not {:?}",
                self.query_within
            ));
        }
        self.local_models().map(drop)
    }

    /// The regions the node embeds through Bedrock in, each once and as
    /// Amazon writes it: a region named twice, or in capitals, or with
    /// space about it, is the one region.
    fn regions(&self) -> Vec<String> {
        let mut regions: Vec<String> = Vec::new();
        for region in self
            .bedrock_regions
            .iter()
            .map(|region| region_named(region))
        {
            if !region.is_empty() && !regions.contains(&region) {
                regions.push(region);
            }
        }
        regions
    }

    /// The names the release lists the node's local models by: its own,
    /// and the others collections may embed with.
    #[cfg(feature = "local")]
    fn local_models(&self) -> std::result::Result<(Option<String>, Vec<String>), String> {
        let listed = |name: &String| match crate::local::named(name) {
            Some(listed) => Ok(listed.model_code.clone()),
            None => Err(format!("{name} is not a local model this release runs")),
        };
        let own = self.local_model.as_ref().map(listed).transpose()?;
        let mut others = Vec::new();
        for name in &self.local_models_allowed {
            let other = listed(name)?;
            if Some(&other) != own.as_ref() && !others.contains(&other) {
                others.push(other);
            }
        }
        // A place is the node's own model's, and the rest are for the
        // others: with no place for any of them, none could ever load.
        let places = self.local_models.max(1) - usize::from(own.is_some());
        if places == 0 && !others.is_empty() {
            // Named as whoever set the node up named them.
            let told = self.local_models_allowed.iter();
            let told: Vec<&str> = { told }
                .filter(|name| listed(name).ok() != own)
                .map(String::as_str)
                .collect();
            return Err(format!(
                "a node with room for one local model has room for its own alone, {}: it \
                 cannot also run {}",
                self.local_model.as_deref().unwrap_or_default(),
                told.join(" and ")
            ));
        }
        Ok((own, others))
    }

    #[cfg(not(feature = "local"))]
    fn local_models(&self) -> std::result::Result<(Option<String>, Vec<String>), String> {
        Ok((None, Vec::new()))
    }
}

/// A region as Amazon writes it, whatever case it was given in and
/// whatever space about it: so that a node and its collections mean the
/// same region by the same name. Nothing else is asked of the name: what
/// regions there are is Amazon's to say.
fn region_named(region: &str) -> String {
    region.trim().to_ascii_lowercase()
}

impl std::fmt::Debug for Settings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Settings")
            .field("cache", &self.cache)
            .field("keys", &self.keys.keys())
            .field("fault_stands", &self.fault_stands)
            .field("query_within", &self.query_within)
            .field("documents_within", &self.documents_within)
            .field("local_models", &self.local_models)
            .field("local_model_idle", &self.local_model_idle)
            .field("local_model", &self.local_model)
            .field("local_models_allowed", &self.local_models_allowed)
            .field("bedrock_regions", &self.bedrock_regions)
            .finish()
    }
}

/// A model as a collection asks for it: where, which, and at what size.
type Asked = (EmbeddingProvider, String, u32);

/// A model as the node runs it or reaches it: what all who embed with it
/// share, whatever prefixes they put before their texts and whichever
/// properties they embed. It is no more than tells one model from
/// another, and no less: models that differ in any of this may differ in
/// whether they can be used.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Run {
    /// A local model, by the name the release lists it by: one model,
    /// loaded once, whichever organisation's copy of it a collection
    /// names. Its size is its own.
    Local(String),
    Bedrock {
        region: String,
        model: String,
        dimensions: u32,
    },
    /// By its URL as the node's keys are given by, so that an endpoint
    /// sent a key is never taken for one that is sent none.
    Http {
        url: String,
        api: HttpApi,
        model: String,
        dimensions: u32,
    },
}

/// What is shared by those who embed with each model, and when each was
/// last asked for.
#[derive(Debug, Default)]
struct Models {
    held: HashMap<Run, (Arc<Shared>, u64)>,
    /// How many times a model has been asked for.
    asked: u64,
}

impl Models {
    /// What is held for `run`, which is thereby the last asked for.
    fn asked_for(&mut self, run: &Run) -> Option<Arc<Shared>> {
        self.asked += 1;
        let (shared, last) = self.held.get_mut(run)?;
        *last = self.asked;
        Some(shared.clone())
    }

    /// Whether there is room to hold something for `run`, made if need be
    /// by forgetting a model served elsewhere: the one asked for longest
    /// ago of those nothing is known of that stands for `stands`, or of
    /// them all if something is known of each.
    ///
    /// A model that is being embedded with is never forgotten. Were it,
    /// the next to ask for it would be given another of what its
    /// embedders share while the first was still in use: two accounts of
    /// whether it can be used, and two of whatever the model counts by.
    /// So when every model kept is in use, there is no room.
    ///
    /// Local models are not counted, and never forgotten: they are no
    /// more than the release lists, each has a thread, and the node
    /// bounds how many are loaded by other means.
    fn room_for(&mut self, run: &Run, stands: Duration) -> bool {
        let elsewhere = |run: &Run| !matches!(run, Run::Local(_));
        if !elsewhere(run) || self.held.keys().filter(|run| elsewhere(run)).count() < ELSEWHERE {
            return true;
        }
        // What is held here and by no one else is in no one's use.
        let longest_ago = self
            .held
            .iter()
            .filter(|(run, (shared, _))| elsewhere(run) && Arc::strong_count(shared) == 1)
            .min_by_key(|(_, (shared, last))| (!shared.quiet(stands), *last))
            .map(|(run, _)| run.clone());
        longest_ago
            .and_then(|forgotten| self.held.remove(&forgotten))
            .is_some()
    }

    /// Hold `shared` for `run`, which is thereby the last asked for.
    fn hold(&mut self, run: Run, shared: Arc<Shared>) {
        self.asked += 1;
        self.held.insert(run, (shared, self.asked));
    }
}

/// The node's embedders. All who embed with one model share it: one
/// loaded model or one client, one set of vectors made for queries, and
/// one account of whether the model can be used.
#[derive(Debug)]
pub struct Embedders {
    settings: Settings,
    models: Mutex<Models>,
    /// The models lately found to be ones this node does not run, so that
    /// each is logged once while it stands.
    refused: Mutex<Told>,
    /// The local models the node has loaded.
    #[cfg(feature = "local")]
    room: Arc<crate::local::Room>,
    /// The local models collections may embed with, by the names the
    /// release lists them by: the node's own first, if it has one.
    #[cfg(feature = "local")]
    allowed: Vec<String>,
    /// The node's calls to Bedrock in each region it embeds through.
    bedrock: HashMap<String, Arc<crate::bedrock::Calls>>,
    /// What every endpoint is reached through, made when the first is.
    endpoints: std::sync::OnceLock<reqwest::Client>,
    /// How many times that was made: once.
    #[cfg(test)]
    endpoint_clients: std::sync::atomic::AtomicUsize,
}

/// What has been said lately, and when: each thing is to be said once in
/// any time a fault stands, however often it comes up.
#[derive(Debug, Default)]
struct Told(HashMap<Asked, Instant>);

impl Told {
    /// Whether `asked` was not spoken of within the time, in which case
    /// it now is: once in each such time, however often it comes up
    /// meanwhile. What was spoken of longer ago is forgotten.
    fn first(&mut self, asked: &Asked, now: Instant, stands: Duration) -> bool {
        self.0.retain(|_, told| now.duration_since(*told) < stands);
        match self.0.contains_key(asked) {
            true => false,
            false => {
                self.0.insert(asked.clone(), now);
                true
            }
        }
    }
}

impl Embedders {
    pub fn new(settings: Settings) -> Self {
        // No time is taken as none at all: a node given no time to wait
        // on a model would refuse everything, and one that kept no model
        // unused would load it for every request.
        let at_least = |given: Duration| given.max(SHORTEST);
        let settings = Settings {
            fault_stands: crate::embedder::standing(settings.fault_stands),
            query_within: at_least(settings.query_within).min(LONGEST_QUERY_WITHIN),
            documents_within: at_least(settings.documents_within),
            local_model_idle: at_least(settings.local_model_idle),
            ..settings
        };
        // What is wrong with how the node is set up is for whoever starts
        // it to refuse: here it is said, and the node runs no local model
        // but its own.
        #[cfg(feature = "local")]
        let (own, others) = settings.local_models().unwrap_or_else(|wrong| {
            tracing::error!(%wrong, "the local models this node is set up to run are left out");
            let own = settings
                .local_model
                .as_deref()
                .and_then(crate::local::named);
            (own.map(|own| own.model_code.clone()), Vec::new())
        });
        #[cfg(feature = "local")]
        {
            let places = settings.local_models.max(1) - usize::from(own.is_some());
            if others.len() > places {
                tracing::warn!(
                    models = others.join(", "),
                    places,
                    "this node is set up to run more local models than it has loaded at once: \
                     collections that embed with one there is no room for wait for another to \
                     go unused"
                );
            }
        }
        // Likewise more regions than a node has calls to share among: it
        // embeds through the first of them.
        // Each once: named twice, a region would have two shares and
        // use one.
        let most = crate::bedrock::MOST_REGIONS;
        let mut regions = settings.regions();
        if regions.len() > most {
            tracing::error!(
                left_out = regions[most..].join(", "),
                "this node is set up to embed through Bedrock in more regions than the {most} \
                 it can"
            );
        }
        regions.truncate(most);
        let settings = Settings {
            bedrock_regions: regions.clone(),
            ..settings
        };
        Self {
            models: Mutex::default(),
            refused: Mutex::default(),
            #[cfg(feature = "local")]
            room: crate::local::Room::new(
                settings.local_models,
                settings.local_model_idle,
                own.clone(),
            ),
            #[cfg(feature = "local")]
            allowed: own.into_iter().chain(others).collect(),
            bedrock: crate::bedrock::Calls::shared_among(&regions),
            endpoints: std::sync::OnceLock::new(),
            #[cfg(test)]
            endpoint_clients: std::sync::atomic::AtomicUsize::new(0),
            settings,
        }
    }

    /// The embedder for a collection that embeds as `embedding` says, or
    /// `None` if its clients supply every vector.
    pub fn embedder(&self, embedding: &Embedding) -> Result<Option<Arc<Embedder>>> {
        let shared = self.shared(embedding);
        // Only a collection there already comes to a refusal here: one
        // made by a release or a build that could run its model, or on a
        // node set up to. It is said once while it stands, however the
        // collections that ask for the model word their texts, and with
        // nothing held.
        if let Err(error) = &shared {
            let first = error.lasting().is_some() && {
                let Embedding {
                    provider,
                    model,
                    dimensions,
                    ..
                } = embedding;
                let asked = (provider.clone(), model.clone(), *dimensions);
                let mut refused = self.refused.lock().expect("refused lock");
                refused.first(&asked, Instant::now(), self.settings.fault_stands)
            };
            if first {
                tracing::error!(
                    model = %embedding.model,
                    cause = %error,
                    "a collection's embedding model cannot be run by this node"
                );
            }
        }
        let settings = &self.settings;
        Ok(shared?.map(|shared| {
            let embedder = Embedder::sharing(shared, embedding.clone())
                .standing_for(settings.fault_stands)
                .answering_within(settings.query_within, settings.documents_within);
            Arc::new(embedder)
        }))
    }

    /// What those who embed as `embedding` says share, held from the
    /// first of them on.
    fn shared(&self, embedding: &Embedding) -> Result<Option<Arc<Shared>>> {
        let Some(run) = self.run(embedding)? else {
            return Ok(None);
        };
        let mut models = self.models.lock().expect("models lock");
        if let Some(shared) = models.asked_for(&run) {
            return Ok(Some(shared));
        }
        if !models.room_for(&run, self.settings.fault_stands) {
            // Which passes as soon as one of them is done with.
            return Err(crate::EmbedError::failed(
                &embedding.model,
                format!(
                    "this node is embedding with {ELSEWHERE} models served elsewhere, \
                     which are as many as it keeps at once"
                ),
            ));
        }
        let shared = Arc::new(Shared::new(self.provider(&run)?));
        models.hold(run, shared.clone());
        Ok(Some(shared))
    }

    /// Whether a collection that embeds as `embedding` says could be
    /// embedded for here, as far as can be told without using the model:
    /// that its model is one of this release's, at that size, and that
    /// this build can run it and this node is set up to. Nothing is
    /// built, started or kept for the asking, so that asking costs
    /// nothing however often and of however many models it is done.
    ///
    /// A model this node is not set up to run is, to whoever asks for a
    /// collection to be made with it, no model to be had here: the
    /// request's fault, as a name the release does not know is. Whether
    /// there is room for the model just now is not asked: that changes
    /// from one moment to the next.
    pub fn admits(&self, embedding: &Embedding) -> Result<()> {
        match self.run(embedding) {
            Err(crate::EmbedError::Unusable { reason, .. }) => {
                Err(crate::EmbedError::NoSuchModel(reason))
            }
            run => run.map(drop),
        }
    }

    /// The model a collection that embeds as `embedding` says is embedded
    /// for with, or `None` if its clients supply every vector: refused if
    /// it is not one this node runs or reaches.
    fn run(&self, embedding: &Embedding) -> Result<Option<Run>> {
        let Embedding {
            provider,
            model,
            dimensions,
            ..
        } = embedding;
        let (model, dimensions) = (model.clone(), *dimensions);
        Ok(Some(match provider {
            EmbeddingProvider::None => return Ok(None),
            EmbeddingProvider::Local => Run::Local(self.runs_locally(&model, dimensions)?),
            EmbeddingProvider::Bedrock { region } => {
                Bedrock::admits(&model, dimensions)?;
                self.calls_in(region, &model)?;
                Run::Bedrock {
                    region: region_named(region),
                    model,
                    dimensions,
                }
            }
            EmbeddingProvider::Http { url, api } => Run::Http {
                url: url.trim_end_matches('/').to_string(),
                api: *api,
                model,
                dimensions,
            },
        }))
    }

    /// The name the release lists `model` by, if it is one it runs at
    /// that size and one this node is set up to run. A model the node is
    /// not set up to run, met in a collection that is there already, is
    /// the node's setup to put right, not the build's: this is the only
    /// model that is refused so before it is ever used.
    #[cfg(feature = "local")]
    fn runs_locally(&self, model: &str, dimensions: u32) -> Result<String> {
        let listed = &crate::local::known(model, dimensions)?.model_code;
        if self.allowed.contains(listed) {
            return Ok(listed.clone());
        }
        // As whoever set the node up named them, which is what they
        // will look for: not as the release lists them.
        // Each once, though it be named again among those allowed, or
        // under another of its names.
        let told = self.settings.local_model.iter();
        let told = { told }.chain(&self.settings.local_models_allowed);
        let mut listed: Vec<Option<&str>> = Vec::new();
        let mut names: Vec<&str> = Vec::new();
        for name in told {
            let model = crate::local::named(name).map(|listed| listed.model_code.as_str());
            if !listed.contains(&model) {
                listed.push(model);
                names.push(name);
            }
        }
        let told = names;
        let runs = match told.is_empty() {
            true => "no model locally".to_string(),
            false => format!("only {} locally", told.join(" and ")),
        };
        Err(crate::EmbedError::unusable(
            model,
            vellev_core::ModelFault::Setup,
            format!("this node is set up to run {runs}, not {model}"),
        ))
    }

    /// The node's calls to Bedrock in `region`, if it is one the node
    /// embeds through. A collection names its own region, and may name
    /// any: one the node is not set up to call is the node's setup to
    /// put right, for a collection that is there already.
    fn calls_in(&self, region: &str, model: &str) -> Result<Arc<crate::bedrock::Calls>> {
        if let Some(calls) = self.bedrock.get(&region_named(region)) {
            return Ok(calls.clone());
        }
        let regions = &self.settings.bedrock_regions;
        let through = match regions.is_empty() {
            true => "in no region".to_string(),
            false => format!("only in {}", regions.join(" and ")),
        };
        Err(crate::EmbedError::unusable(
            model,
            vellev_core::ModelFault::Setup,
            format!("this node is set up to embed through Bedrock {through}, not in {region}"),
        ))
    }

    #[cfg(not(feature = "local"))]
    fn runs_locally(&self, model: &str, _dimensions: u32) -> Result<String> {
        Err(Self::without_local(model))
    }

    /// What a build without local models says of one.
    #[cfg_attr(feature = "local", cfg(test))]
    fn without_local(model: &str) -> crate::EmbedError {
        crate::EmbedError::Unsupported(format!("this build cannot run {model} locally"))
    }

    /// How many times what endpoints are reached through was made.
    #[cfg(test)]
    fn endpoint_clients_made(&self) -> usize {
        self.endpoint_clients
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    /// How many local models are loaded, or loading, each taking memory
    /// of its own: none where local models are not built in.
    pub fn local_models_loaded(&self) -> usize {
        #[cfg(feature = "local")]
        return self.room.count();
        #[cfg(not(feature = "local"))]
        0
    }

    /// How many models have what their embedders share held.
    pub fn held(&self) -> usize {
        self.models.lock().expect("models lock").held.len()
    }

    fn provider(&self, run: &Run) -> Result<Arc<dyn Provider>> {
        Ok(match run {
            Run::Local(listed) => self.local(listed)?,
            Run::Bedrock {
                region,
                model,
                dimensions,
            } => {
                let calls = self.calls_in(region, model)?;
                Arc::new(Bedrock::among(region, model, *dimensions, calls)?)
            }
            Run::Http {
                url, api, model, ..
            } => {
                let token = self.settings.keys.get(url).cloned();
                let client = match self.endpoints.get() {
                    Some(client) => client.clone(),
                    None => {
                        let client = Http::client(model)?;
                        #[cfg(test)]
                        self.endpoint_clients
                            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        self.endpoints.get_or_init(|| client).clone()
                    }
                };
                Arc::new(Http::through(client, url, *api, model, token)?)
            }
        })
    }

    /// The model the release lists as `listed`, loaded when it is used
    /// and there is room for it.
    #[cfg(feature = "local")]
    fn local(&self, listed: &str) -> Result<Arc<dyn Provider>> {
        let model = crate::local::named(listed).ok_or_else(|| {
            crate::EmbedError::NoSuchModel(format!("{listed} cannot be run locally"))
        })?;
        let cache = &self.settings.cache;
        Ok(Arc::new(crate::Local::among(
            model,
            cache,
            self.room.clone(),
        )?))
    }

    #[cfg(not(feature = "local"))]
    fn local(&self, listed: &str) -> Result<Arc<dyn Provider>> {
        Err(Self::without_local(listed))
    }
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use vellev_core::ModelFault;

    use super::*;
    use crate::{EmbedError, Purpose};

    fn embedding(provider: EmbeddingProvider, model: &str, dimensions: u32) -> Embedding {
        Embedding {
            provider,
            model: model.into(),
            dimensions,
            query_prefix: None,
            document_prefix: None,
            properties: Vec::new(),
        }
    }

    fn through(url: &str, model: &str) -> Embedding {
        let provider = EmbeddingProvider::Http {
            url: url.into(),
            api: HttpApi::Tei,
        };
        embedding(provider, model, 2)
    }

    #[tokio::test(start_paused = true)]
    async fn a_thing_is_said_once_while_it_stands_and_then_forgotten() {
        let asked = |model: &str| (EmbeddingProvider::Local, model.to_string(), 2);
        let mut told = Told::default();
        assert!(told.first(&asked("one"), Instant::now(), FAULT_STANDS));
        assert!(!told.first(&asked("one"), Instant::now(), FAULT_STANDS));
        assert!(told.first(&asked("two"), Instant::now(), FAULT_STANDS));
        // Coming up all the while, it is said once in each such time: it
        // is not put off for having come up.
        let step = FAULT_STANDS / 4;
        let mut said = 1;
        for _ in 0..12 {
            tokio::time::advance(step).await;
            said += usize::from(told.first(&asked("one"), Instant::now(), FAULT_STANDS));
        }
        assert_eq!(said, 1 + 3, "once at first, and once in each time after");
        assert_eq!(
            told.0.len(),
            1,
            "what was last spoken of long ago is forgotten"
        );
    }

    #[test]
    fn a_fault_is_taken_to_stand_no_less_than_a_second_and_no_more_than_an_hour() {
        let taken = |fault_stands| {
            let settings = Settings {
                fault_stands,
                ..Settings::default()
            };
            Embedders::new(settings).settings.fault_stands
        };
        assert_eq!(taken(Duration::ZERO), Duration::from_secs(1));
        assert_eq!(taken(Duration::from_millis(999)), Duration::from_secs(1));
        assert_eq!(taken(FAULT_STANDS), FAULT_STANDS);
        assert_eq!(
            taken(Duration::from_secs(3_601)),
            Duration::from_secs(3_600)
        );
        assert_eq!(taken(Duration::MAX), Duration::from_secs(3_600));
    }

    #[test]
    fn a_build_without_local_models_is_at_fault_not_whoever_asks_for_one() {
        let said = Embedders::without_local("BAAI/bge-base-en-v1.5");
        assert_eq!(
            said,
            crate::EmbedError::Unsupported(
                "this build cannot run BAAI/bge-base-en-v1.5 locally".into()
            )
        );
        assert_eq!(said.lasting(), Some(ModelFault::Build));
    }

    /// Whether what those who embed as each says share is one and the
    /// same.
    fn share(embedders: &Embedders, one: &Embedding, other: &Embedding) -> bool {
        let shared = |embedding| embedders.shared(embedding).unwrap().unwrap();
        Arc::ptr_eq(&shared(one), &shared(other))
    }

    /// How a node is set up that has `own` for its own local model and
    /// may run `others` too.
    #[cfg(feature = "local")]
    fn running(own: Option<&str>, others: &[&str]) -> Settings {
        Settings {
            local_model: own.map(String::from),
            local_models_allowed: others.iter().map(|name| name.to_string()).collect(),
            ..Settings::default()
        }
    }

    /// A node set up to embed through Bedrock in these regions.
    fn calling(regions: &[&str]) -> Embedders {
        Embedders::new(Settings {
            bedrock_regions: regions.iter().map(|region| region.to_string()).collect(),
            ..Settings::default()
        })
    }

    #[cfg(feature = "local")]
    #[test]
    fn a_local_model_is_held_once_however_it_is_named_and_however_texts_are_worded() {
        let also = running(Some("BAAI/bge-base-en-v1.5"), &["BAAI/bge-small-en-v1.5"]);
        let embedders = Embedders::new(also);
        let named = |model: &str| embedding(EmbeddingProvider::Local, model, 768);
        let first = named("BAAI/bge-base-en-v1.5");
        let worded = Embedding {
            query_prefix: Some("Represent this sentence: ".into()),
            document_prefix: Some("passage: ".into()),
            properties: vec!["title".into(), "body".into()],
            ..first.clone()
        };
        let names = [
            "Xenova/bge-base-en-v1.5",
            "bge-base-en-v1.5",
            "BGE-Base-EN-v1.5",
            "anyone-at-all/bge-base-en-v1.5",
        ];
        for other in names.map(named).iter().chain([&worded]) {
            assert!(share(&embedders, &first, other), "{other:?}");
        }
        assert_eq!(embedders.held(), 1);
        // Another model is another: though it too is held once.
        let small = embedding(EmbeddingProvider::Local, "BAAI/bge-small-en-v1.5", 384);
        assert!(!share(&embedders, &first, &small));
        assert_eq!(embedders.held(), 2);
        // Each embedder is still the collection's own way of embedding.
        let embedder = embedders.embedder(&worded).unwrap().unwrap();
        assert_eq!(embedder.embedding(), &worded);
    }

    #[cfg(feature = "local")]
    #[test]
    fn a_node_runs_the_local_models_it_is_set_up_to_and_its_own_alone_unless_told() {
        let named =
            |model: &str, dimensions| embedding(EmbeddingProvider::Local, model, dimensions);
        let base = named("BAAI/bge-base-en-v1.5", 768);
        let small = named("BAAI/bge-small-en-v1.5", 384);
        // What is said of a model the node is not set up to run: to
        // whoever asks for a collection to be made with it, that there is
        // no such model to be had, which is the request's fault; and of a
        // collection there already, that the node's setup is at fault.
        let refused = |embedders: &Embedders, embedding: &Embedding, said: &str| {
            let made = embedders.admits(embedding);
            assert_eq!(made, Err(EmbedError::NoSuchModel(said.into())));
            let used = embedders.embedder(embedding).map(drop);
            let setup = EmbedError::Unusable {
                model: embedding.model.clone(),
                reason: said.into(),
                fault: ModelFault::Setup,
            };
            assert_eq!(used, Err(setup));
        };

        // Told nothing, a node runs its own model and no other of the
        // forty-odd the release lists.
        let untold = Settings::default();
        assert_eq!(untold.local_model.as_deref(), Some(DEFAULT_LOCAL_MODEL));
        assert!(untold.local_models_allowed.is_empty());
        assert_eq!(untold.checked(), Ok(()));
        let embedders = Embedders::new(untold);
        assert_eq!(embedders.admits(&base), Ok(()));
        assert!(embedders.embedder(&base).is_ok());
        let only = "this node is set up to run only BAAI/bge-base-en-v1.5 locally, \
                    not BAAI/bge-small-en-v1.5";
        refused(&embedders, &small, only);
        assert_eq!(embedders.held(), 1, "nothing is held for what is refused");
        let own = |embedders: &Embedders| embedders.room.set_up().2.map(String::from);
        assert_eq!(own(&embedders).as_deref(), Some("Xenova/bge-base-en-v1.5"));

        // Told others, it runs those too, named by any organisation's
        // copy: and its own is still its own, wherever the others are
        // named and whether or not it is named among them.
        let also = running(
            Some("bge-base-en-v1.5"),
            &[
                "BAAI/bge-small-en-v1.5",
                "bge-small-en-v1.5",
                "BAAI/bge-base-en-v1.5",
            ],
        );
        assert_eq!(also.checked(), Ok(()));
        let embedders = Embedders::new(also);
        // What it runs is said as it was told, and each model once,
        // though it was named again, or by another of its names.
        let large = named("BAAI/bge-large-en-v1.5", 1024);
        let these = "this node is set up to run only bge-base-en-v1.5 and \
                     BAAI/bge-small-en-v1.5 locally, not BAAI/bge-large-en-v1.5";
        refused(&embedders, &large, these);
        // Its own named again first among those allowed: said once.
        let again = running(Some("bge-base-en-v1.5"), &["BAAI/bge-base-en-v1.5"]);
        let embedders = Embedders::new(again);
        let once = "this node is set up to run only bge-base-en-v1.5 locally, not \
                    BAAI/bge-large-en-v1.5";
        refused(&embedders, &large, once);
        let embedders = Embedders::new(running(
            Some("bge-base-en-v1.5"),
            &[
                "BAAI/bge-small-en-v1.5",
                "bge-small-en-v1.5",
                "BAAI/bge-base-en-v1.5",
            ],
        ));
        let both = ["Xenova/bge-base-en-v1.5", "Xenova/bge-small-en-v1.5"];
        assert_eq!(embedders.allowed, both);
        assert_eq!(own(&embedders).as_deref(), Some("Xenova/bge-base-en-v1.5"));
        for model in ["Xenova/bge-small-en-v1.5", "bge-small-en-v1.5"] {
            assert_eq!(embedders.admits(&named(model, 384)), Ok(()));
            assert!(embedders.embedder(&named(model, 384)).is_ok());
        }
        assert_eq!(embedders.admits(&base), Ok(()));
        // What is no model at all is still that, whoever asks and when.
        let unheard_of = named("acme/unheard-of", 768);
        let no_such = EmbedError::NoSuchModel("acme/unheard-of cannot be run locally".into());
        assert_eq!(embedders.admits(&unheard_of), Err(no_such.clone()));
        assert_eq!(embedders.embedder(&unheard_of).map(drop), Err(no_such));
        // Models served elsewhere are not local ones.
        assert_eq!(embedders.admits(&through("http://one.test", "any")), Ok(()));

        // Another model for its own: that one is kept, and the model a
        // node runs unless told otherwise is not run unless it is named
        // too. What it runs is said as it was told, not as the release
        // lists it.
        let embedders = Embedders::new(running(Some("bge-small-en-v1.5"), &[]));
        assert_eq!(own(&embedders).as_deref(), Some("Xenova/bge-small-en-v1.5"));
        assert_eq!(embedders.admits(&small), Ok(()));
        let only = "this node is set up to run only bge-small-en-v1.5 locally, \
                    not BAAI/bge-base-en-v1.5";
        refused(&embedders, &base, only);
        // No model of its own: none is kept, and it runs what it is told.
        let embedders = Embedders::new(running(None, &["bge-small-en-v1.5"]));
        assert_eq!(own(&embedders), None);
        assert_eq!(embedders.admits(&small), Ok(()));
        // And none at all.
        let embedders = Embedders::new(running(None, &[]));
        let none = "this node is set up to run no model locally, not BAAI/bge-base-en-v1.5";
        refused(&embedders, &base, none);
    }

    /// What is wrong with how a node is set up to run local models is
    /// for whoever starts it to be told, and not found when a
    /// collection that should have worked does not.
    #[cfg(feature = "local")]
    #[test]
    fn a_node_set_up_to_run_what_it_cannot_is_told_so_before_it_starts() {
        let wrong = |settings: Settings| settings.checked().unwrap_err();
        // A slip in a name, its own or another's: not left out quietly,
        // to leave the node running without the model it was to keep.
        assert_eq!(
            wrong(running(Some("BAAI/bge-base-en-v15"), &[])),
            "BAAI/bge-base-en-v15 is not a local model this release runs"
        );
        assert_eq!(
            wrong(running(
                Some("bge-base-en-v1.5"),
                &["bge-small-en-v1.5", "acme/unheard-of"]
            )),
            "acme/unheard-of is not a local model this release runs"
        );
        // Room for one model is room for the node's own: another could
        // never be loaded, and every collection made with it would wait
        // for ever.
        let one = |own, others: &[&str]| Settings {
            local_models: 1,
            ..running(own, others)
        };
        assert_eq!(
            wrong(one(Some("bge-base-en-v1.5"), &["bge-small-en-v1.5"])),
            "a node with room for one local model has room for its own alone, \
             bge-base-en-v1.5: it cannot also run bge-small-en-v1.5"
        );
        assert_eq!(
            wrong(one(Some("bge-base-en-v1.5"), &["bge-small-en-v1.5"])),
            wrong(Settings {
                local_models: 0,
                ..running(Some("bge-base-en-v1.5"), &["bge-small-en-v1.5"])
            })
        );
        // Its own named again among the others is no other.
        assert_eq!(
            one(Some("bge-base-en-v1.5"), &["BAAI/bge-base-en-v1.5"]).checked(),
            Ok(())
        );
        assert_eq!(
            one(None, &["bge-small-en-v1.5", "bge-base-en-v1.5"]).checked(),
            Ok(())
        );
        // More than there are places for, though some could load, is
        // the node's to run as it is told: each waits its turn.
        let more = running(
            Some("bge-base-en-v1.5"),
            &["bge-small-en-v1.5", "bge-large-en-v1.5"],
        );
        assert_eq!(more.checked(), Ok(()));

        // A node started regardless runs its own model, and leaves out
        // what it was wrongly told.
        let regardless = running(
            Some("bge-base-en-v1.5"),
            &["bge-small-en-v1.5", "acme/unheard-of"],
        );
        let embedders = Embedders::new(regardless);
        assert_eq!(embedders.allowed, ["Xenova/bge-base-en-v1.5"]);
    }

    /// A collection names the region it embeds through Bedrock in, and
    /// may name any, of Amazon's or of its own making. Only those the
    /// node is set up to call are called.
    #[test]
    fn a_collection_embeds_through_bedrock_only_in_the_regions_the_node_is_set_up_to_call() {
        let titan = |region: &str| {
            let region = EmbeddingProvider::Bedrock {
                region: region.into(),
            };
            embedding(region, "amazon.titan-embed-text-v2:0", 256)
        };
        let refused = |embedders: &Embedders, region: &str, said: &str| {
            let made = embedders.admits(&titan(region));
            assert_eq!(made, Err(EmbedError::NoSuchModel(said.into())), "{region}");
            let used = embedders.embedder(&titan(region)).map(drop);
            let setup = EmbedError::Unusable {
                model: "amazon.titan-embed-text-v2:0".into(),
                reason: said.into(),
                fault: ModelFault::Setup,
            };
            assert_eq!(used, Err(setup), "{region}");
        };
        // Told of none, a node calls none.
        assert!(Settings::default().bedrock_regions.is_empty());
        let embedders = Embedders::new(Settings::default());
        let none = "this node is set up to embed through Bedrock in no region, not in eu-west-2";
        refused(&embedders, "eu-west-2", none);

        // Any region it is told of, however Amazon names it.
        let embedders = calling(&["eu-west-2", "eusc-de-east-1"]);
        for region in ["eu-west-2", "eusc-de-east-1"] {
            assert_eq!(embedders.admits(&titan(region)), Ok(()));
            assert!(embedders.embedder(&titan(region)).is_ok());
        }
        // In whatever case and with whatever space about it a collection
        // writes the region, it is that region, and the one model.
        for region in ["EU-WEST-2", " eu-west-2 ", "Eu-West-2"] {
            assert_eq!(embedders.admits(&titan(region)), Ok(()), "{region:?}");
            assert!(embedders.embedder(&titan(region)).is_ok(), "{region:?}");
        }
        assert_eq!(embedders.held(), 2, "one model in each region");
        // And no other: a real one, one made up to look like one, or
        // none at all.
        for region in ["us-east-1", "zz-nowhere-1", "eu-west-22", ""] {
            let said = format!(
                "this node is set up to embed through Bedrock only in eu-west-2 and \
                 eusc-de-east-1, not in {region}"
            );
            refused(&embedders, region, &said);
        }
        assert_eq!(embedders.held(), 2, "nothing is held for what is refused");
        // A region named twice is one region, with one share of the
        // calls, however it is written each time; and a blank is none.
        let twice = calling(&["eu-west-2", " EU-West-2", "", "eusc-de-east-1", "eu-west-2"]);
        assert_eq!(
            twice.settings.bedrock_regions,
            ["eu-west-2", "eusc-de-east-1"]
        );
        assert_eq!(twice.bedrock.len(), 2);
        // A node has calls to share among five regions and no more: told
        // of six, it is to be refused when it starts, and started
        // regardless embeds through the first five.
        let six = [
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "eu-north-1",
            "eu-south-1",
            "us-east-1",
        ];
        let told = |regions: &[&str]| Settings {
            bedrock_regions: regions.iter().map(|region| region.to_string()).collect(),
            ..Settings::default()
        };
        assert_eq!(told(&six[..5]).checked(), Ok(()));
        // The regions are counted as the node will call them: five named
        // eight times over are five.
        let eight = [&six[..5], &["EU-WEST-1", " eu-west-2 ", "eu-south-1"][..]].concat();
        assert_eq!(told(&eight).checked(), Ok(()));
        assert_eq!(calling(&eight).bedrock.len(), 5);
        assert_eq!(
            told(&six).checked().unwrap_err(),
            "a node embeds through Bedrock in 5 regions at the most, each with a share of \
             its calls to itself, and not in the 6 of eu-west-1, eu-west-2, eu-west-3, \
             eu-north-1, eu-south-1, us-east-1"
        );
        let regardless = calling(&six);
        assert_eq!(regardless.admits(&titan("eu-south-1")), Ok(()));
        assert!(regardless.admits(&titan("us-east-1")).is_err());

        // What is no model of Bedrock's is still that, whatever the
        // region.
        let made_up = Embedding {
            model: "amazon.titan-embed-text-v2:0-mine".into(),
            ..titan("zz-nowhere-1")
        };
        let no_such = embedders.admits(&made_up).unwrap_err().to_string();
        assert!(
            no_such.contains("is not amazon.titan-embed-text-v2"),
            "{no_such}"
        );
    }

    #[cfg(feature = "local")]
    #[test]
    fn a_node_has_room_for_as_many_local_models_as_it_is_set_up_to_and_one_at_the_least() {
        let seconds = Duration::from_secs;
        let told = |local_models, local_model_idle| {
            let embedders = Embedders::new(Settings {
                local_models,
                local_model_idle,
                ..Settings::default()
            });
            let (most, idle, own) = embedders.room.set_up();
            (most, idle, own.map(String::from))
        };
        let own = || Some("Xenova/bge-base-en-v1.5".to_string());
        assert_eq!((LOCAL_MODELS, LOCAL_MODEL_IDLE), (2, seconds(900)));
        assert_eq!(
            told(LOCAL_MODELS, LOCAL_MODEL_IDLE),
            (2, seconds(900), own())
        );
        assert_eq!(told(5, seconds(60)), (5, seconds(60), own()));
        assert_eq!(told(0, Duration::ZERO), (1, seconds(1), own()));
    }

    /// A key for an endpoint is sixteen characters or more and has no
    /// space in it, or the node does not start: said by its endpoint,
    /// never by the key.
    #[test]
    fn a_key_too_short_to_be_one_is_refused_before_the_node_starts() {
        assert_eq!(SHORTEST_KEY, 16);
        let keyed = |key: &str| Settings {
            keys: [("http://tei:80".to_string(), key.to_string())].into(),
            ..Settings::default()
        };
        for short in [
            "1",
            "40",
            "e",
            "ab",
            "the",
            "0123456789abcde",
            "sk-abcdef 0123456789",
        ] {
            let refused = keyed(short).checked().unwrap_err();
            assert_eq!(
                refused,
                "the key for http://tei:80 is shorter than 16 characters or has space in it, \
                 which no key an endpoint gives out is: leave the key out if the endpoint \
                 needs none"
            );
        }
        assert_eq!(keyed("0123456789abcdef").checked(), Ok(()));
        assert_eq!(Settings::default().checked(), Ok(()));
    }

    /// A model is given a second to answer at the least, and a search is
    /// kept waiting two minutes at the most: a node told more is to be
    /// refused when it starts, and started regardless keeps to the most.
    #[test]
    fn a_model_is_given_a_second_to_answer_at_the_least_and_a_search_two_minutes_at_the_most() {
        // How long a query and a call for documents are given, by a node
        // told these.
        let given = |query_within, documents_within| {
            let settings = Embedders::new(Settings {
                query_within,
                documents_within,
                ..Settings::default()
            })
            .settings;
            [settings.query_within, settings.documents_within]
        };
        let (second, ms) = (Duration::from_secs(1), Duration::from_millis(999));
        assert_eq!(given(Duration::ZERO, ms), [second; 2]);
        assert_eq!(given(second * 3, second * 7), [second * 3, second * 7]);
        assert_eq!(LONGEST_QUERY_WITHIN, second * 120);
        assert_eq!(
            given(second * 120, second * 900),
            [second * 120, second * 900]
        );
        assert_eq!(given(second * 121, second), [second * 120, second]);
        let told = |query_within| Settings {
            query_within,
            ..Settings::default()
        };
        assert_eq!(told(second * 120).checked(), Ok(()));
        assert_eq!(
            told(second * 121).checked(),
            Err(
                "a search is kept waiting for its text's vector for 120s at the most, not 121s"
                    .to_string()
            )
        );
    }

    /// A model that never answers, and has an activity of its own to say
    /// what it is doing by, which it never does.
    #[derive(Debug, Default)]
    struct Stopped(crate::Activity);

    #[async_trait]
    impl Provider for Stopped {
        async fn embed(&self, _: Vec<String>, _: Purpose) -> Result<Vec<Vec<f32>>> {
            std::future::pending().await
        }

        fn activity(&self) -> Option<&crate::Activity> {
            Some(&self.0)
        }
    }

    /// The times a node is told are those its embedders keep to.
    #[tokio::test(start_paused = true)]
    async fn an_embedder_gives_a_model_the_times_its_node_is_told() {
        let seconds = Duration::from_secs;
        let embedders = Embedders::new(Settings {
            query_within: seconds(3),
            documents_within: seconds(7),
            ..Settings::default()
        });
        let asked = through("http://one.test", "stopped");
        let run = embedders.run(&asked).unwrap().unwrap();
        let stopped = Arc::new(Shared::new(Arc::new(Stopped::default())));
        embedders.models.lock().unwrap().hold(run, stopped);
        let embedder = embedders.embedder(&asked).unwrap().unwrap();
        let said = |made: EmbedError| made.to_string();

        let started = Instant::now();
        let query = embedder.query("fox").await.unwrap_err();
        assert_eq!(started.elapsed(), seconds(3));
        assert!(said(query).ends_with("no answer within 3s"));
        let started = Instant::now();
        let documents = embedder.documents(&["fox".to_string()]).await.unwrap_err();
        assert_eq!(started.elapsed(), seconds(7));
        assert!(said(documents).ends_with("the model has made nothing for anyone in 7s"));
    }

    /// Making what endpoints are reached through reads the certificates
    /// they are trusted by, which is slow: a node makes one, and reaches
    /// every endpoint, model and size through it.
    #[test]
    fn every_endpoint_is_reached_through_one_client() {
        let embedders = calling(&[]);
        assert_eq!(embedders.endpoint_clients_made(), 0);
        for (url, model) in [
            ("http://one.test", "m"),
            ("http://two.test", "m"),
            ("http://one.test", "n"),
        ] {
            embedders.embedder(&through(url, model)).unwrap().unwrap();
        }
        let sized = Embedding {
            dimensions: 3,
            ..through("http://one.test", "m")
        };
        embedders.embedder(&sized).unwrap().unwrap();
        assert_eq!(embedders.held(), 4);
        assert_eq!(embedders.endpoint_clients_made(), 1);
    }

    #[test]
    fn those_who_embed_alike_through_an_endpoint_share_and_no_others_do() {
        let embedders = calling(&["eu-west-2", "us-east-1"]);
        let first = through("http://one.test", "m");
        let worded = Embedding {
            query_prefix: Some("find: ".into()),
            properties: vec!["title".into()],
            ..first.clone()
        };
        assert!(share(&embedders, &first, &worded));
        assert!(share(&embedders, &first, &through("http://one.test/", "m")));
        assert_eq!(embedders.held(), 1);

        // Whatever may make one usable and another not is not shared
        // across: the endpoint, how it is spoken to, the model, its size,
        // and the key, which goes by the URL as it is given.
        let spoken = EmbeddingProvider::Http {
            url: "http://one.test".into(),
            api: HttpApi::Openai,
        };
        let others = [
            through("http://two.test", "m"),
            through("http://one.test/v1", "m"),
            through("http://one.test", "n"),
            embedding(first.provider.clone(), "m", 3),
            embedding(spoken, "m", 2),
        ];
        for other in &others {
            assert!(!share(&embedders, &first, other), "{other:?}");
        }
        assert_eq!(embedders.held(), 1 + others.len());

        let region = |region: &str| EmbeddingProvider::Bedrock {
            region: region.into(),
        };
        let titan =
            |region, dimensions| embedding(region, "amazon.titan-embed-text-v2:0", dimensions);
        let (here, there) = (
            titan(region("eu-west-2"), 256),
            titan(region("us-east-1"), 256),
        );
        assert!(share(&embedders, &here, &here.clone()));
        assert!(!share(&embedders, &here, &there));
        assert!(!share(&embedders, &here, &titan(region("eu-west-2"), 512)));
    }

    /// A model that refuses every call for good.
    #[derive(Debug)]
    struct Refusing;

    #[async_trait]
    impl Provider for Refusing {
        async fn embed(&self, _: Vec<String>, _: Purpose) -> Result<Vec<Vec<f32>>> {
            Err(EmbedError::Unusable {
                model: "m".into(),
                reason: "no such model".into(),
                fault: ModelFault::Provider,
            })
        }
    }

    #[tokio::test(start_paused = true)]
    async fn no_more_models_served_elsewhere_are_kept_than_a_node_keeps() {
        let embedders = Embedders::new(Settings::default());
        let numbered = |n: usize| through("http://one.test", &format!("model {n}"));
        // What is held for a model, known by where it is, and let go of
        // at once: it is in no one's use.
        let asked = |n: usize| Arc::as_ptr(&embedders.shared(&numbered(n)).unwrap().unwrap());

        // More models than are kept, asked for one after another, with
        // the first of them asked for all the while.
        let first = asked(0);
        for n in 1..ELSEWHERE + 20 {
            asked(n);
            assert_eq!(asked(0), first, "{n}");
            assert!(embedders.held() <= ELSEWHERE);
        }
        assert_eq!(embedders.held(), ELSEWHERE);
        // Those asked for last are the ones kept.
        let runs = |n: usize| embedders.run(&numbered(n)).unwrap().unwrap();
        let held = |n: usize| {
            let models = embedders.models.lock().unwrap();
            models.held.contains_key(&runs(n))
        };
        assert!(held(0) && held(ELSEWHERE + 19) && held(21));
        assert!(!held(1) && !held(20));

        // A model with a fault that stands is not forgotten while there
        // is one without: forgotten, it would be asked and refused, and
        // said to be at fault, all over again.
        let models = || embedders.models.lock().unwrap();
        let run = runs(ELSEWHERE + 20);
        let refusing = {
            let refusing = Arc::new(Shared::new(Arc::new(Refusing)));
            assert!(models().room_for(&run, FAULT_STANDS));
            models().hold(run.clone(), refusing.clone());
            let embedder = Embedder::sharing(refusing.clone(), numbered(ELSEWHERE + 20));
            assert!(embedder.query("fox").await.is_err());
            Arc::downgrade(&refusing)
        };
        let quiet = || refusing.upgrade().map(|shared| shared.quiet(FAULT_STANDS));
        assert_eq!(quiet(), Some(false));
        for n in 100..100 + 2 * ELSEWHERE {
            asked(n);
        }
        assert_eq!(embedders.held(), ELSEWHERE);
        assert_eq!(quiet(), Some(false), "it is still held");
        // Once it stands no longer, it is forgotten like any other.
        tokio::time::advance(FAULT_STANDS).await;
        assert_eq!(quiet(), Some(true));
        for n in 300..300 + ELSEWHERE {
            asked(n);
        }
        assert_eq!(quiet(), None);
        assert_eq!(embedders.held(), ELSEWHERE);
    }

    /// Forgotten while it was in use, a model would be held afresh for
    /// the next to ask for it, beside the one still in use: and whatever
    /// its embedders share to count by, such as a fault that stands,
    /// would be counted twice over.
    #[test]
    fn a_model_served_elsewhere_is_not_forgotten_while_it_is_embedded_with() {
        let embedders = Embedders::new(Settings::default());
        let numbered = |n: usize| through("http://one.test", &format!("model {n}"));
        let shared = |n: usize| embedders.shared(&numbered(n));

        // One model in use, and many times more than are kept asked for
        // and let go of meanwhile: it is the same one all along.
        let in_use = embedders.embedder(&numbered(0)).unwrap().unwrap();
        let first = shared(0).unwrap().unwrap();
        for n in 1..4 * ELSEWHERE {
            drop(shared(n).unwrap());
        }
        assert!(Arc::ptr_eq(&first, &shared(0).unwrap().unwrap()));
        assert_eq!(embedders.held(), ELSEWHERE);
        drop((in_use, first));

        // As many in use as are kept: there is no room for one more,
        // which is refused in a way that may pass, and passes as soon as
        // one of them is done with. None is forgotten for it.
        let embedders = Embedders::new(Settings::default());
        let shared = |n: usize| embedders.shared(&numbered(n));
        let mut in_use: Vec<_> = (0..ELSEWHERE)
            .map(|n| shared(n).unwrap().unwrap())
            .collect();
        let refused = shared(ELSEWHERE).unwrap_err();
        assert_eq!(
            refused,
            EmbedError::Failed {
                model: format!("model {ELSEWHERE}"),
                reason: "this node is embedding with 64 models served elsewhere, which are \
                         as many as it keeps at once"
                    .into(),
            }
        );
        assert_eq!(refused.lasting(), None);
        for (n, held) in in_use.iter().enumerate() {
            assert!(Arc::ptr_eq(held, &shared(n).unwrap().unwrap()), "{n}");
        }
        // Those kept can all still be asked for, and a local model too.
        #[cfg(feature = "local")]
        {
            let local = embedding(EmbeddingProvider::Local, "BAAI/bge-base-en-v1.5", 768);
            assert!(embedders.shared(&local).is_ok());
        }
        in_use.truncate(ELSEWHERE - 1);
        assert!(shared(ELSEWHERE).is_ok());
        assert!(
            shared(ELSEWHERE - 1).is_ok(),
            "it was forgotten, and is held afresh"
        );
    }

    #[cfg(feature = "local")]
    #[test]
    fn local_models_are_not_forgotten_to_make_room_for_those_served_elsewhere() {
        let embedders = Embedders::new(Settings::default());
        let local = embedding(EmbeddingProvider::Local, "BAAI/bge-base-en-v1.5", 768);
        let first = embedders.shared(&local).unwrap().unwrap();
        for n in 0..2 * ELSEWHERE {
            let numbered = through("http://one.test", &format!("model {n}"));
            embedders.shared(&numbered).unwrap().unwrap();
        }
        assert_eq!(embedders.held(), ELSEWHERE + 1);
        let still = embedders.shared(&local).unwrap().unwrap();
        assert!(Arc::ptr_eq(&first, &still));
    }
}
