use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

use crate::{Object, Value};

/// What an object's content is, as a reader is told it and a writer may
/// be asked to find it still: a hash of the properties and the vector,
/// written as 32 hexadecimal characters. Two objects of the same content
/// have the same version wherever and whenever they are read, and a
/// change to either, to the vector alone included, gives another.
///
/// A client compares versions for equality only: they are not ordered,
/// and nothing is to be read into them. An object changed and changed
/// back has the version it had; a write conditional on it then finds the
/// content it read, exactly, which is what the condition is for.
///
/// How the hash is made is a compatibility contract between releases,
/// kept in docs/compatibility.md: a node of one release and a node of
/// the next give the same version to the same content.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "openapi", schema(value_type = String))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Version([u8; 16]);

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("a version is 32 hexadecimal characters, not {0:?}")]
pub struct VersionError(String);

/// What the hash is over, first, so that it is of nothing else.
const DOMAIN: &[u8] = b"vellev object version 1\n";

/// A number's bits, with the one zero: the writer takes `-0.0` to be
/// `0.0` when it asks whether a write changes anything, and so does the
/// version.
fn f64_bits(n: f64) -> [u8; 8] {
    if n == 0.0 { 0.0_f64 } else { n }.to_le_bytes()
}

fn f32_bits(n: f32) -> [u8; 4] {
    if n == 0.0 { 0.0_f32 } else { n }.to_le_bytes()
}

fn text(hasher: &mut Sha256, text: &str) {
    hasher.update((text.len() as u64).to_le_bytes());
    hasher.update(text.as_bytes());
}

impl Version {
    /// The version of an object of these `properties` and this `vector`.
    pub fn of(properties: &BTreeMap<String, Value>, vector: Option<&[f32]>) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(DOMAIN);
        hasher.update((properties.len() as u64).to_le_bytes());
        for (name, value) in properties {
            text(&mut hasher, name);
            match value {
                Value::Text(value) => {
                    hasher.update([1]);
                    text(&mut hasher, value);
                }
                Value::TextArray(values) => {
                    hasher.update([2]);
                    hasher.update((values.len() as u64).to_le_bytes());
                    for value in values {
                        text(&mut hasher, value);
                    }
                }
                Value::Int(n) => {
                    hasher.update([3]);
                    hasher.update(n.to_le_bytes());
                }
                Value::Number(n) => {
                    hasher.update([4]);
                    hasher.update(f64_bits(*n));
                }
                Value::Bool(b) => {
                    hasher.update([5]);
                    hasher.update([u8::from(*b)]);
                }
                Value::Date(at) => {
                    hasher.update([6]);
                    hasher.update(at.timestamp().to_le_bytes());
                    hasher.update(at.timestamp_subsec_nanos().to_le_bytes());
                }
            }
        }
        match vector {
            None => hasher.update([0]),
            Some(vector) => {
                hasher.update([1]);
                hasher.update((vector.len() as u64).to_le_bytes());
                // Fed a block at a time: a hasher given four bytes at a
                // time spends longer taking them in than hashing them.
                let mut block = [0_u8; 1_024];
                for numbers in vector.chunks(block.len() / 4) {
                    for (bytes, n) in block.chunks_exact_mut(4).zip(numbers) {
                        bytes.copy_from_slice(&f32_bits(*n));
                    }
                    hasher.update(&block[..numbers.len() * 4]);
                }
            }
        }
        let digest = hasher.finalize();
        let mut version = [0; 16];
        version.copy_from_slice(&digest[..16]);
        Self(version)
    }

    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

impl Object {
    /// The object's version: that of its properties and vector.
    pub fn version(&self) -> Version {
        Version::of(&self.properties, self.vector.as_deref())
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in &self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Version({self})")
    }
}

impl std::str::FromStr for Version {
    type Err = VersionError;

    /// The 32 hexadecimal characters a version is written as, in either
    /// case.
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let malformed = || VersionError(text.to_string());
        if text.len() != 32 || !text.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(malformed());
        }
        let mut version = [0; 16];
        for (byte, pair) in version.iter_mut().zip(text.as_bytes().chunks(2)) {
            let pair = std::str::from_utf8(pair).map_err(|_| malformed())?;
            *byte = u8::from_str_radix(pair, 16).map_err(|_| malformed())?;
        }
        Ok(Self(version))
    }
}

impl Serialize for Version {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for Version {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = <std::borrow::Cow<'de, str>>::deserialize(deserializer)?;
        text.parse().map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};

    use super::*;
    use crate::ObjectId;

    fn object(properties: &[(&str, Value)], vector: Option<Vec<f32>>) -> Object {
        Object {
            id: ObjectId(uuid::Uuid::from_u128(7)),
            properties: properties
                .iter()
                .map(|(name, value)| (name.to_string(), value.clone()))
                .collect(),
            vector,
        }
    }

    /// These versions are what they are for good: a release that gives
    /// another to the same content breaks every version a client holds.
    /// Each was computed again outside this crate from the encoding
    /// written in docs/compatibility.md.
    #[test]
    fn a_version_is_what_the_contract_says_it_is() {
        let golden = [
            (object(&[], None), "6ff3b35929f2ef8827ef084b967ea53a"),
            (
                object(&[], Some(vec![1.0, -0.0, 0.5])),
                "9c53675940bb3bc20e54ea089d361361",
            ),
            (
                object(
                    &[
                        ("title", Value::Text("A red fox".into())),
                        ("n", Value::Int(-3)),
                        ("score", Value::Number(0.25)),
                        ("ok", Value::Bool(true)),
                        ("tags", Value::TextArray(vec!["a".into(), "bé".into()])),
                        (
                            "at",
                            Value::Date(Utc.with_ymd_and_hms(2026, 10, 9, 12, 30, 15).unwrap()),
                        ),
                    ],
                    Some(vec![0.1, 0.2]),
                ),
                "08b86decc967c85cc066c6794319d606",
            ),
            (
                object(&[("n", Value::Int(1))], None),
                "6d03f05d07799d71e6a72c9e5aef4fbd",
            ),
            (
                object(
                    &[
                        ("n", Value::Int(1)),
                        (
                            "at",
                            Value::Date(
                                Utc.with_ymd_and_hms(2026, 10, 9, 12, 30, 15).unwrap()
                                    + chrono::Duration::milliseconds(250),
                            ),
                        ),
                    ],
                    None,
                ),
                "9c9ebb0010a3b02ae9aa939cbd1c1761",
            ),
        ];
        for (object, version) in golden {
            assert_eq!(object.version().to_string(), version, "{object:?}");
        }
    }

    #[test]
    fn the_version_is_the_contents_and_nothing_elses() {
        let a = object(&[("n", Value::Int(1))], Some(vec![1.0, 2.0]));
        let same = Object {
            id: ObjectId(uuid::Uuid::from_u128(8)),
            ..a.clone()
        };
        assert_eq!(a.version(), same.version(), "the id is not covered");
        let differing = [
            object(&[("n", Value::Int(2))], Some(vec![1.0, 2.0])),
            object(&[("n", Value::Int(1))], Some(vec![1.0, 2.5])),
            object(&[("n", Value::Int(1))], Some(vec![1.0])),
            object(&[("n", Value::Int(1))], None),
            object(&[("n", Value::Number(1.0))], Some(vec![1.0, 2.0])),
            object(&[("n", Value::Text("1".into()))], Some(vec![1.0, 2.0])),
            object(&[("m", Value::Int(1))], Some(vec![1.0, 2.0])),
            object(
                &[("n", Value::Int(1)), ("m", Value::Int(1))],
                Some(vec![1.0, 2.0]),
            ),
            object(&[], Some(vec![1.0, 2.0])),
        ];
        for other in differing {
            assert_ne!(a.version(), other.version(), "{other:?}");
        }
        // The content is compared as the writer compares it: either zero
        // is the one zero, and a date is its instant.
        let zeros = [
            object(&[("x", Value::Number(0.0))], Some(vec![0.0])),
            object(&[("x", Value::Number(-0.0))], Some(vec![-0.0])),
        ];
        assert_eq!(zeros[0].version(), zeros[1].version());
        assert_eq!(zeros[0].properties, zeros[1].properties);
        assert_eq!(zeros[0].vector, zeros[1].vector);
        let at = Utc.with_ymd_and_hms(2026, 10, 9, 12, 30, 15).unwrap();
        let elsewhere = at.with_timezone(&chrono::FixedOffset::east_opt(3_600).unwrap());
        assert_eq!(
            object(&[("at", Value::Date(at))], None).version(),
            object(&[("at", Value::Date(elsewhere.with_timezone(&Utc)))], None).version()
        );
        // Text arrays are not texts, nor one text their join.
        assert_ne!(
            object(
                &[("t", Value::TextArray(vec!["a".into(), "b".into()]))],
                None
            )
            .version(),
            object(&[("t", Value::TextArray(vec!["ab".into()]))], None).version()
        );
        assert_ne!(
            object(&[("t", Value::TextArray(vec!["a".into()]))], None).version(),
            object(&[("t", Value::Text("a".into()))], None).version()
        );
    }

    #[test]
    fn a_version_is_written_as_hex_and_read_back_in_either_case() {
        let version = object(&[("n", Value::Int(1))], None).version();
        let written = version.to_string();
        assert_eq!(written.len(), 32);
        assert!(written
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()));
        assert_eq!(written.parse::<Version>(), Ok(version));
        assert_eq!(written.to_uppercase().parse::<Version>(), Ok(version));
        assert_eq!(
            serde_json::to_value(version).unwrap(),
            serde_json::Value::String(written.clone())
        );
        assert_eq!(
            serde_json::from_value::<Version>(serde_json::Value::String(written.clone())).unwrap(),
            version
        );
        // A sign is not a digit, though a number parser takes one.
        let signed = format!("+{}", &written[1..]);
        let signed_later = format!("{}-{}", &written[..20], &written[21..]);
        for bad in [
            "",
            "abc",
            &written[..31],
            &format!("{written}0"),
            &format!("g{}", &written[1..]),
            "é".repeat(16).as_str(),
            signed.as_str(),
            signed_later.as_str(),
            " ".repeat(32).as_str(),
        ] {
            assert!(bad.parse::<Version>().is_err(), "{bad:?}");
            assert!(
                serde_json::from_value::<Version>(serde_json::Value::String(bad.to_string()))
                    .is_err()
            );
        }
        assert!(serde_json::from_str::<Version>("12").is_err());
        assert_eq!(format!("{version:?}"), format!("Version({written})"));
    }
}
