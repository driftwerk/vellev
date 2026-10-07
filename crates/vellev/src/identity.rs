//! Who is asking, inside a tenant, and so which rows of a collection with
//! an access rule they may read.
//!
//! A caller with a tenant's key names the person it asks for with a token
//! in [`IDENTITY`]: a JWT signed with Ed25519 by whoever authenticated them,
//! naming the tenant, the person's identity id and their teams, and when
//! it expires. vellev holds the public keys it trusts by key id, so a key
//! is rotated by adding the new one, switching the signer, and removing the
//! old once its tokens have expired. A collection's access rule is then
//! added to every read of it: a row is read only by its owner or a member
//! of the team it names. A tenant's key without a token reads nothing of a
//! ruled collection; an admin key reads everything.

use std::collections::HashMap;

use axum::extract::{FromRequestParts, RawPathParams};
use axum::http::request::Parts;
use eyre::{bail, eyre, Result};
use jsonwebtoken::{Algorithm, DecodingKey, Validation};
use serde::Deserialize;
use serde_json::json;
use vellev_api::{Code, IDENTITY};
use vellev_core::{AccessRule, Collection, Object};
use vellev_index::filter::Filter;

use crate::problem::ApiError;
use crate::server::App;

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
            .map(|given| {
                let (kid, x) = given
                    .split_once('=')
                    .ok_or_else(|| eyre!("an identity key is KID=X, not {given}"))?;
                let key = DecodingKey::from_ed_components(x)
                    .map_err(|error| eyre!("identity key {kid}: {error}"))?;
                if kid.is_empty() {
                    bail!("an identity key needs an id");
                }
                Ok((kid.to_string(), key))
            })
            .collect::<Result<_>>()
            .map(Self)
    }

    /// The person `token` names, if it is signed by a trusted key, has not
    /// expired, and is for `tenant`.
    fn verify(&self, token: &str, tenant: &str) -> std::result::Result<Identity, ApiError> {
        let invalid = |why: &str| {
            ApiError::new(
                Code::Unauthorized,
                format!("the identity token is not valid: {why}"),
            )
        };
        let header = jsonwebtoken::decode_header(token).map_err(|_| invalid("unreadable"))?;
        let key = header
            .kid
            .as_ref()
            .and_then(|kid| self.0.get(kid))
            .ok_or_else(|| invalid("signed by a key not trusted"))?;
        let mut validation = Validation::new(Algorithm::EdDSA);
        validation.set_required_spec_claims(&["exp"]);
        let claims = jsonwebtoken::decode::<Claims>(token, key, &validation)
            .map_err(|error| invalid(&error.to_string()))?
            .claims;
        if claims.tenant != tenant {
            return Err(invalid("for another tenant"));
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Caller {
    Admin,
    Tenant,
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
        if parts.extensions.get::<Caller>() != Some(&Caller::Tenant) {
            return Ok(Viewer::Everything);
        }
        let Some(token) = parts.headers.get(IDENTITY) else {
            return Ok(Viewer::Nobody);
        };
        let token = token
            .to_str()
            .map_err(|_| ApiError::invalid(format!("{IDENTITY} is not text")))?
            .to_string();
        let tenant = RawPathParams::from_request_parts(parts, app)
            .await
            .ok()
            .and_then(|params| {
                params
                    .iter()
                    .find(|(name, _)| *name == "tenant")
                    .map(|(_, tenant)| tenant.to_string())
            })
            .unwrap_or_default();
        app.identity_keys
            .verify(&token, &tenant)
            .map(Viewer::Person)
    }
}
