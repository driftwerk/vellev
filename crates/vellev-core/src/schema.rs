use serde::{Deserialize, Serialize};

use crate::CollectionName;

/// Largest vector a collection may declare. Well above any embedding model in
/// use (Titan v2 is 1024, OpenAI's largest is 3072) and small enough that a
/// typo cannot reserve gigabytes per object.
pub const MAX_DIMENSIONS: u32 = 8192;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SchemaError {
    #[error("property name {0:?} must be letters, digits or '_', starting with a letter or '_'")]
    PropertyName(String),
    #[error("property {0:?} is declared twice")]
    DuplicateProperty(String),
    #[error("property {0:?} is {1}, and only text and text[] can be searchable")]
    SearchableNotText(String, DataType),
    #[error("embedded property {0:?} is not declared")]
    EmbeddedUnknown(String),
    #[error("embedded property {0:?} is {1}, and only text and text[] can be embedded")]
    EmbeddedNotText(String, DataType),
    #[error("embedding dimensions must be between 1 and {MAX_DIMENSIONS}, got {0}")]
    Dimensions(u32),
    #[error("embedding model id must not be empty")]
    Model,
    #[error("bm25 needs k1 >= 0 and 0 <= b <= 1, got k1 = {k1}, b = {b}")]
    Bm25 { k1: f32, b: f32 },
    #[error("access rule property {0:?} is not declared")]
    AccessUnknown(String),
    #[error("access rule property {0:?} must be filterable text")]
    AccessNotFilterableText(String),
    #[error("property {0:?} already exists with a different definition")]
    PropertyConflict(String),
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataType {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "text[]")]
    TextArray,
    #[serde(rename = "int")]
    Int,
    #[serde(rename = "number")]
    Number,
    #[serde(rename = "bool")]
    Bool,
    #[serde(rename = "date")]
    Date,
}

impl DataType {
    pub fn is_text(self) -> bool {
        matches!(self, DataType::Text | DataType::TextArray)
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            DataType::Text => "text",
            DataType::TextArray => "text[]",
            DataType::Int => "int",
            DataType::Number => "number",
            DataType::Bool => "bool",
            DataType::Date => "date",
        })
    }
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Property {
    pub name: String,
    pub data_type: DataType,
    /// Indexed for filters. A filter on text matches the whole value exactly;
    /// it is never tokenized.
    #[serde(default = "yes")]
    pub filterable: bool,
    /// Tokenized and indexed for keyword search.
    #[serde(default)]
    pub searchable: bool,
}

fn yes() -> bool {
    true
}

impl Property {
    pub fn new(name: &str, data_type: DataType) -> Self {
        Self {
            name: name.to_string(),
            data_type,
            filterable: true,
            searchable: false,
        }
    }

    pub fn searchable(mut self) -> Self {
        self.searchable = true;
        self
    }
}

/// What an embedding endpoint speaks.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HttpApi {
    /// A text-embeddings-inference server's `/embed`.
    #[default]
    Tei,
    /// An OpenAI-compatible `/v1/embeddings`.
    Openai,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "snake_case")]
pub enum EmbeddingProvider {
    /// A model run in the node's own process.
    Local,
    Bedrock {
        region: String,
    },
    Http {
        url: String,
        #[serde(default)]
        api: HttpApi,
    },
    /// The client supplies every vector.
    None,
}

/// Why a node cannot use a collection's embedding model until its setup is
/// put right: where the fault lies, and so who is to put it right.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ModelFault {
    /// The node's build cannot run the model: another build may.
    Build,
    /// The node is set up wrong for the model: it may not write where the
    /// model is kept, or what serves the model does not know its key.
    Setup,
    /// What serves the model says it has no such model, or gives vectors
    /// that are not the collection's size: the collection names a model
    /// that is not there to be had.
    Provider,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Embedding {
    #[serde(flatten)]
    pub provider: EmbeddingProvider,
    /// Recorded even when the client supplies vectors, so a query is never
    /// embedded with a different model from the one that produced the data.
    pub model: String,
    pub dimensions: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_prefix: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document_prefix: Option<String>,
    /// The properties whose text is embedded, in order.
    #[serde(default)]
    pub properties: Vec<String>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Distance {
    #[default]
    Cosine,
    Dot,
    L2,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Bm25 {
    pub k1: f32,
    pub b: f32,
}

impl Default for Bm25 {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

/// Who may read a row. Enforced by the node on every query, so a caller that
/// forgets to filter sees nothing extra.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccessRule {
    /// A row is readable when this property equals the caller's identity id.
    pub owner: String,
    /// A row is also readable when this property equals `team_prefix` followed
    /// by the id of a team the caller belongs to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub team: Option<String>,
    #[serde(default)]
    pub team_prefix: String,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Collection {
    pub name: CollectionName,
    pub properties: Vec<Property>,
    pub embedding: Embedding,
    #[serde(default)]
    pub distance: Distance,
    #[serde(default)]
    pub bm25: Bm25,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub access_rule: Option<AccessRule>,
}

impl Collection {
    pub fn property(&self, name: &str) -> Option<&Property> {
        self.properties.iter().find(|p| p.name == name)
    }

    pub fn validate(&self) -> Result<(), SchemaError> {
        for (i, property) in self.properties.iter().enumerate() {
            validate_property(property)?;
            if self.properties[..i].iter().any(|p| p.name == property.name) {
                return Err(SchemaError::DuplicateProperty(property.name.clone()));
            }
        }
        let e = &self.embedding;
        if e.dimensions == 0 || e.dimensions > MAX_DIMENSIONS {
            return Err(SchemaError::Dimensions(e.dimensions));
        }
        if e.model.trim().is_empty() {
            return Err(SchemaError::Model);
        }
        for name in &e.properties {
            let property = self
                .property(name)
                .ok_or_else(|| SchemaError::EmbeddedUnknown(name.clone()))?;
            if !property.data_type.is_text() {
                return Err(SchemaError::EmbeddedNotText(
                    name.clone(),
                    property.data_type,
                ));
            }
        }
        let Bm25 { k1, b } = self.bm25;
        if !(k1 >= 0.0 && k1.is_finite() && (0.0..=1.0).contains(&b)) {
            return Err(SchemaError::Bm25 { k1, b });
        }
        if let Some(rule) = &self.access_rule {
            for name in std::iter::once(&rule.owner).chain(&rule.team) {
                let property = self
                    .property(name)
                    .ok_or_else(|| SchemaError::AccessUnknown(name.clone()))?;
                if property.data_type != DataType::Text || !property.filterable {
                    return Err(SchemaError::AccessNotFilterableText(name.clone()));
                }
            }
        }
        Ok(())
    }

    /// Additive migration. Returns whether the property was added; adding one
    /// that already exists unchanged is a no-op, so migrations can be re-run.
    pub fn add_property(&mut self, property: Property) -> Result<bool, SchemaError> {
        validate_property(&property)?;
        match self.property(&property.name) {
            Some(existing) if *existing == property => Ok(false),
            Some(_) => Err(SchemaError::PropertyConflict(property.name)),
            None => {
                self.properties.push(property);
                Ok(true)
            }
        }
    }
}

fn validate_property(property: &Property) -> Result<(), SchemaError> {
    let mut chars = property.name.chars();
    let first_ok = chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
    if !first_ok || !chars.all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(SchemaError::PropertyName(property.name.clone()));
    }
    if property.searchable && !property.data_type.is_text() {
        return Err(SchemaError::SearchableNotText(
            property.name.clone(),
            property.data_type,
        ));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use DataType::*;

    fn collection(name: &str, embedding: Embedding, properties: Vec<Property>) -> Collection {
        Collection {
            name: CollectionName::new(name).unwrap(),
            properties,
            embedding,
            distance: Distance::Cosine,
            bm25: Bm25::default(),
            access_rule: None,
        }
    }

    fn bge(properties: &[&str]) -> Embedding {
        Embedding {
            provider: EmbeddingProvider::Local,
            model: "BAAI/bge-base-en-v1.5".into(),
            dimensions: 768,
            query_prefix: Some("Represent this sentence for searching relevant passages: ".into()),
            document_prefix: None,
            properties: properties.iter().map(|p| p.to_string()).collect(),
        }
    }

    /// Workforce's memory collection as driftwerk runs it.
    pub(crate) fn workforce_memory() -> Collection {
        let mut c = collection(
            "WorkforceMemory_bge_base_en_v1_5",
            bge(&["content"]),
            vec![
                Property::new("content", Text).searchable(),
                Property::new("memoryId", Text),
                Property::new("layer", Text),
                Property::new("owner", Text),
                Property::new("visibility", Text),
                Property::new("importance", Number),
                Property::new("tags", TextArray),
                Property::new("deleted", Bool),
                Property::new("accessCount", Int),
                Property::new("createdAt", Date),
                Property::new("lastAccessed", Date),
                Property::new("source", Text),
                Property::new("identityId", Text),
                Property::new("sourceSessionId", Text),
                Property::new("sourceMessageIds", TextArray),
                Property::new("recurrenceCount", Int),
                Property::new("lastRecurrenceAt", Date),
                Property::new("eventTime", Date),
            ],
        );
        c.access_rule = Some(AccessRule {
            owner: "identityId".into(),
            team: Some("visibility".into()),
            team_prefix: "team:".into(),
        });
        c
    }

    fn workforce_collections() -> Vec<Collection> {
        let titan = Embedding {
            provider: EmbeddingProvider::Bedrock {
                region: "eu-west-2".into(),
            },
            model: "amazon.titan-embed-text-v2:0".into(),
            dimensions: 1024,
            query_prefix: None,
            document_prefix: None,
            properties: vec!["summaryText".into()],
        };
        let marker = Embedding {
            provider: EmbeddingProvider::None,
            model: "none".into(),
            dimensions: 1,
            query_prefix: None,
            document_prefix: None,
            properties: vec![],
        };
        vec![
            workforce_memory(),
            collection(
                "WorkforceMemorySwitch",
                marker,
                vec![
                    Property::new("collection", Text),
                    Property::new("switchedAt", Date),
                ],
            ),
            collection(
                "CodeEntity",
                bge(&["searchableContent"]),
                vec![
                    Property::new("searchableContent", Text).searchable(),
                    Property::new("qualifiedName", Text).searchable(),
                    Property::new("name", Text).searchable(),
                    Property::new("signature", Text).searchable(),
                    Property::new("docComment", Text).searchable(),
                    Property::new("summary", Text).searchable(),
                    Property::new("entityId", Text),
                    Property::new("repoId", Text),
                    Property::new("kind", Text),
                    Property::new("filePath", Text),
                    Property::new("language", Text),
                    Property::new("startLine", Int),
                    Property::new("endLine", Int),
                ],
            ),
            collection(
                "CodeSummary",
                titan,
                vec![
                    Property::new("summaryText", Text).searchable(),
                    Property::new("entityId", Text),
                    Property::new("repoId", Text),
                    Property::new("granularity", Text),
                    Property::new("filePath", Text),
                    Property::new("isStale", Bool),
                ],
            ),
            collection(
                "Artifacts",
                bge(&["text"]),
                vec![
                    Property::new("artifactId", Text),
                    Property::new("text", Text).searchable(),
                ],
            ),
        ]
    }

    #[test]
    fn every_workforce_collection_is_expressible() {
        for c in workforce_collections() {
            assert_eq!(c.validate(), Ok(()), "{}", c.name);
        }
    }

    #[test]
    fn a_schema_survives_a_json_round_trip() {
        for c in workforce_collections() {
            let json = serde_json::to_string(&c).unwrap();
            assert_eq!(serde_json::from_str::<Collection>(&json).unwrap(), c);
        }
    }

    #[test]
    fn a_schema_written_by_a_newer_release_reads_without_what_it_added() {
        let newer = serde_json::json!({
            "name": "notes",
            "properties": [{ "name": "title", "data_type": "text", "searchable": true, "shown": false }],
            "embedding": { "provider": "none", "model": "client", "dimensions": 2 },
            "bm25": { "k1": 1.2, "b": 0.75, "boost": 2 },
            "access_rule": { "owner": "owner", "audit": true },
            "replicas": 3
        });
        let read: Collection = serde_json::from_value(newer).unwrap();
        assert_eq!(read.properties[0].name, "title");
        assert_eq!(read.bm25, Bm25::default());
        assert_eq!(read.access_rule.unwrap().owner, "owner");
    }

    #[test]
    fn a_schema_reads_from_the_json_a_client_would_send() {
        let c: Collection = serde_json::from_value(serde_json::json!({
            "name": "Notes",
            "properties": [
                { "name": "body", "data_type": "text", "searchable": true },
                { "name": "tags", "data_type": "text[]" }
            ],
            "embedding": {
                "provider": "http", "url": "http://embeddings:80",
                "model": "bge-base-en-v1.5", "dimensions": 768, "properties": ["body"]
            }
        }))
        .unwrap();
        assert_eq!(c.validate(), Ok(()));
        assert!(c.property("tags").unwrap().filterable);
        assert!(!c.property("tags").unwrap().searchable);
        assert_eq!(c.bm25, Bm25 { k1: 1.2, b: 0.75 });
        assert_eq!(c.distance, Distance::Cosine);
    }

    #[test]
    fn invalid_schemas_are_refused_with_the_reason() {
        let base = workforce_memory;
        type Break = fn(&mut Collection);
        let cases: Vec<(Break, SchemaError)> = vec![
            (
                |c| c.properties.push(Property::new("tags", Text)),
                SchemaError::DuplicateProperty("tags".into()),
            ),
            (
                |c| c.properties.push(Property::new("a-b", Text)),
                SchemaError::PropertyName("a-b".into()),
            ),
            (
                |c| c.properties.push(Property::new("", Text)),
                SchemaError::PropertyName("".into()),
            ),
            (
                |c| c.properties.push(Property::new("n", Int).searchable()),
                SchemaError::SearchableNotText("n".into(), Int),
            ),
            (
                |c| c.embedding.properties.push("nope".into()),
                SchemaError::EmbeddedUnknown("nope".into()),
            ),
            (
                |c| c.embedding.properties.push("importance".into()),
                SchemaError::EmbeddedNotText("importance".into(), Number),
            ),
            (|c| c.embedding.dimensions = 0, SchemaError::Dimensions(0)),
            (
                |c| c.embedding.dimensions = 8193,
                SchemaError::Dimensions(8193),
            ),
            (|c| c.embedding.model = " ".into(), SchemaError::Model),
            (|c| c.bm25.b = 1.5, SchemaError::Bm25 { k1: 1.2, b: 1.5 }),
            (
                |c| c.access_rule.as_mut().unwrap().owner = "nobody".into(),
                SchemaError::AccessUnknown("nobody".into()),
            ),
            (
                |c| c.access_rule.as_mut().unwrap().team = Some("tags".into()),
                SchemaError::AccessNotFilterableText("tags".into()),
            ),
        ];
        for (break_it, expected) in cases {
            let mut c = base();
            break_it(&mut c);
            assert_eq!(c.validate(), Err(expected));
        }
    }

    #[test]
    fn an_access_rule_cannot_rest_on_an_unindexed_property() {
        let mut c = workforce_memory();
        c.properties
            .iter_mut()
            .find(|p| p.name == "identityId")
            .unwrap()
            .filterable = false;
        assert_eq!(
            c.validate(),
            Err(SchemaError::AccessNotFilterableText("identityId".into()))
        );
    }

    #[test]
    fn adding_a_property_is_additive_and_repeatable() {
        let mut c = workforce_memory();
        let before = c.properties.len();
        assert_eq!(c.add_property(Property::new("reviewedAt", Date)), Ok(true));
        assert_eq!(c.add_property(Property::new("reviewedAt", Date)), Ok(false));
        assert_eq!(c.properties.len(), before + 1);
        assert_eq!(
            c.add_property(Property::new("reviewedAt", Text)),
            Err(SchemaError::PropertyConflict("reviewedAt".into()))
        );
        assert_eq!(
            c.add_property(Property::new("bad name", Text)),
            Err(SchemaError::PropertyName("bad name".into()))
        );
        assert_eq!(c.validate(), Ok(()));
    }
}
