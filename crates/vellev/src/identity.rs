//! Who is asking, inside a tenant, and so which rows of a collection with
//! an access rule they may read.
//!
//! A caller with a tenant's key names the person it asks for with a token
//! in [`IDENTITY`]: a JWT for the audience [`AUDIENCE`], signed with Ed25519
//! by whoever authenticated them, naming the tenant, the person's identity
//! id and their teams, and when it expires. vellev holds the public keys it trusts by key id, so a key
//! is rotated by adding the new one, switching the signer, and removing the
//! old once its tokens have expired. A collection's access rule is then
//! added to every read of it: a row is read only by its owner or a member
//! of the team it names. A tenant's key without a token reads nothing of a
//! ruled collection; an admin key reads everything. Writes are not
//! narrowed: a tenant's key is trusted with every write, as the indexer
//! must be, and what a write answers says something of the rows it touches.

use std::collections::HashMap;

use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use eyre::{bail, eyre, Result};
use jsonwebtoken::{Algorithm, DecodingKey, Validation};
use serde::Deserialize;
use serde_json::json;
use vellev_api::{Code, IDENTITY};
use vellev_core::{AccessRule, Collection, Object, TenantId};
use vellev_index::filter::Filter;

use crate::problem::ApiError;
use crate::server::App;

/// The audience an identity token must be for.
pub const AUDIENCE: &str = "vellev";
/// The most a token may be, in bytes, and the most teams it may name: each
/// team is a term in every read's filter.
const MOST_TOKEN_BYTES: usize = 8 * 1024;
const MOST_TEAMS: usize = 256;

/// The keys whose tokens are trusted, by key id.
#[derive(Default)]
pub struct IdentityKeys(HashMap<String, DecodingKey>);

impl std::fmt::Debug for IdentityKeys {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.0.keys()).finish()
    }
}

impl IdentityKeys {
    /// Keys given as `KID=X`, X being the Ed25519 public key's 32 bytes in
    /// unpadded base64url, as a JWK's `x` holds them.
    pub fn parse(given: &[String]) -> Result<Self> {
        given
            .iter()
            .try_fold(Self::default(), |Self(mut keys), given| {
                let (kid, x) = given
                    .split_once('=')
                    .map(|(kid, x)| (kid.trim(), x.trim()))
                    .ok_or_else(|| eyre!("an identity key is KID=X, not {given}"))?;
                if kid.is_empty() {
                    bail!("an identity key needs an id");
                }
                let bytes = URL_SAFE_NO_PAD
                    .decode(x)
                    .map_err(|error| eyre!("identity key {kid}: {error}"))?;
                if bytes.len() != 32 {
                    bail!(
                        "identity key {kid} is {} bytes, not an Ed25519 key's 32",
                        bytes.len()
                    );
                }
                let key = DecodingKey::from_ed_components(x)
                    .map_err(|error| eyre!("identity key {kid}: {error}"))?;
                if keys.insert(kid.to_string(), key).is_some() {
                    bail!("identity key {kid} is given twice");
                }
                Ok(Self(keys))
            })
    }

    /// The person `token` names, if it is signed by a trusted key, has not
    /// expired, and is for `tenant`.
    fn verify(&self, token: &str, tenant: &TenantId) -> std::result::Result<Identity, ApiError> {
        let invalid = |why: &str| {
            ApiError::new(
                Code::Unauthorized,
                format!("the identity token is not valid: {why}"),
            )
        };
        if token.len() > MOST_TOKEN_BYTES {
            return Err(invalid("too long"));
        }
        let header = jsonwebtoken::decode_header(token).map_err(|_| invalid("unreadable"))?;
        let key = header
            .kid
            .as_ref()
            .and_then(|kid| self.0.get(kid))
            .ok_or_else(|| invalid("signed by a key not trusted"))?;
        let mut validation = Validation::new(Algorithm::EdDSA);
        validation.set_required_spec_claims(&["exp", "aud"]);
        validation.set_audience(&[AUDIENCE]);
        let claims = jsonwebtoken::decode::<Claims>(token, key, &validation)
            .map_err(|error| invalid(&error.to_string()))?
            .claims;
        if claims.tenant != tenant.as_str() {
            return Err(invalid("for another tenant"));
        }
        if claims.identity_id.is_empty() || claims.teams.iter().any(String::is_empty) {
            return Err(invalid("names nobody"));
        }
        if claims.teams.len() > MOST_TEAMS {
            return Err(invalid("names too many teams"));
        }
        Ok(Identity {
            id: claims.identity_id,
            teams: claims.teams,
        })
    }
}

#[derive(Deserialize)]
struct Claims {
    tenant: String,
    identity_id: String,
    #[serde(default)]
    teams: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Identity {
    pub id: String,
    pub teams: Vec<String>,
}

/// What kind of key a request came with, as the server found it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Caller {
    Admin,
    /// A tenant's key, reaching this tenant.
    Tenant(TenantId),
}

/// Whose eyes a read is for.
#[derive(Debug, Clone, PartialEq)]
pub enum Viewer {
    /// An admin's: every row.
    Everything,
    /// One person's: the rows each collection's rule lets them read.
    Person(Identity),
    /// A tenant's key with no person named: no row of a ruled collection.
    Nobody,
}

impl Viewer {
    /// `filter`, narrowed to the rows of `schema` the viewer may read.
    pub fn narrowed(&self, schema: &Collection, filter: Option<Filter>) -> Option<Filter> {
        let Some(rule) = self.rule(schema) else {
            return filter;
        };
        Some(match filter {
            Some(filter) => Filter::And(vec![rule, filter]),
            None => rule,
        })
    }

    /// Whether the viewer may read `object` of `schema`.
    pub fn sees(&self, schema: &Collection, object: &Object) -> bool {
        self.rule(schema).is_none_or(|rule| {
            rule.compile(schema)
                .is_ok_and(|predicate| predicate.matches(object))
        })
    }

    /// The filter of what the viewer may read of `schema`, if it is ruled.
    fn rule(&self, schema: &Collection) -> Option<Filter> {
        let AccessRule {
            owner,
            team,
            team_prefix,
        } = schema.access_rule.as_ref()?;
        Some(match self {
            Viewer::Everything => return None,
            Viewer::Nobody => Filter::Or(Vec::new()),
            Viewer::Person(person) => {
                let owns = Filter::Eq {
                    property: owner.clone(),
                    value: json!(person.id),
                };
                let shares = team.iter().flat_map(|team| {
                    person.teams.iter().map(move |id| Filter::Eq {
                        property: team.clone(),
                        value: json!(format!("{team_prefix}{id}")),
                    })
                });
                Filter::Or(std::iter::once(owns).chain(shares).collect())
            }
        })
    }
}

impl FromRequestParts<App> for Viewer {
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, app: &App) -> Result<Self, ApiError> {
        let tenant = match parts.extensions.get::<Caller>() {
            Some(Caller::Admin) => return Ok(Viewer::Everything),
            Some(Caller::Tenant(tenant)) => tenant,
            None => {
                return Err(ApiError::new(
                    Code::Internal,
                    "a read reached without its caller known",
                ))
            }
        };
        let Some(token) = parts.headers.get(IDENTITY) else {
            return Ok(Viewer::Nobody);
        };
        let token = token
            .to_str()
            .map_err(|_| ApiError::invalid(format!("{IDENTITY} is not text")))?;
        app.identity_keys.verify(token, tenant).map(Viewer::Person)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::server::Limits;

    #[tokio::test]
    async fn a_read_whose_caller_is_not_known_is_refused() {
        let app = App::new(
            Arc::from([]),
            Limits {
                results: 10,
                body: 1024,
                embedding_urls: Arc::from([]),
                reads: 4,
                reads_wait: std::time::Duration::ZERO,
            },
        );
        let (mut parts, ()) = axum::http::Request::new(()).into_parts();
        let refused = Viewer::from_request_parts(&mut parts, &app).await;
        assert!(refused.is_err());
    }

    const X: &str = "11qYAYKxCrfVS_7TyWQHOg7hcvPapiMlrwIaaPcHURo";

    fn parsed(given: &[&str]) -> Result<IdentityKeys> {
        IdentityKeys::parse(
            &given
                .iter()
                .map(|given| given.to_string())
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn keys_are_given_as_an_id_and_an_ed25519_key_each_once() {
        let keys = parsed(&[&format!("a={X}"), &format!(" b = {X}")]).unwrap();
        assert_eq!(format!("{keys:?}").len(), r#"["a", "b"]"#.len());
        for (given, why) in [
            (X.to_string(), "is KID=X"),
            (format!("={X}"), "needs an id"),
            ("a=not base64!".into(), "identity key a"),
            ("a=AAAA".into(), "3 bytes"),
        ] {
            let refused = parsed(&[&given]).unwrap_err().to_string();
            assert!(refused.contains(why), "{given}: {refused}");
        }
        let twice = parsed(&[&format!("a={X}"), &format!("a={X}")]).unwrap_err();
        assert!(twice.to_string().contains("given twice"), "{twice}");
    }
}
