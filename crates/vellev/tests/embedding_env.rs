//! What a node makes of `VELLEV_BEDROCK_REGIONS` in its environment: in a
//! test binary of its own, the environment being the whole process's.

use clap::Parser;
use vellev::config::Config;

#[derive(Parser)]
struct Parsed {
    #[command(flatten)]
    config: Config,
}

fn parsed() -> Config {
    let required = ["vellev", "--database-url", "x", "--store", "memory://"];
    Parsed::try_parse_from(required).unwrap().config
}

/// Set but blank, in the environment as on the command line, the setting
/// is given and names no region: the node embeds through Bedrock in
/// none, its own region included. Only unset does it fall to the node's
/// own region. `config::tests` shows what each parsed value comes to.
#[test]
fn bedrock_regions_blank_in_the_environment_is_no_region_and_not_the_nodes_own() {
    std::env::remove_var("VELLEV_BEDROCK_REGIONS");
    assert_eq!(parsed().bedrock_regions, None, "unset: the node's own");

    std::env::set_var("VELLEV_BEDROCK_REGIONS", "");
    let blank = parsed().bedrock_regions.expect("given, though blank");
    assert!(
        blank.iter().all(|region| region.trim().is_empty()),
        "{blank:?}"
    );

    std::env::set_var("VELLEV_BEDROCK_REGIONS", "eu-west-2, us-east-1");
    let named = parsed().bedrock_regions.expect("given");
    assert_eq!(named, ["eu-west-2", " us-east-1"]);
}
