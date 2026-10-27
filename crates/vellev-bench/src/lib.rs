//! The nightly benchmark of vellev against Weaviate (design r2): both
//! engines on one runner, on a fixed public corpus, compared at recall
//! targets stated before the first run.

pub mod docker;
pub mod engine;
pub mod fixture;
pub mod measure;
pub mod report;
pub mod run;
