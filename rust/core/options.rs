// rust/core/options.rs
pub fn default_option_value(spec: &str) -> String {
    let Some((_, values)) = spec.split_once("; ") else {
        return spec.to_owned();
    };
    values.split('|').next().unwrap_or_default().to_owned()
}

pub fn override_option(key: &str, default_value: &str) -> String {
    match key {
        "mupen64plus-rdp-plugin" => "angrylion".to_owned(),
        _ => default_value.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::default_option_value;

    #[test]
    fn default_option_value_returns_first_choice() {
        assert_eq!(
            default_option_value("Renderer; angrylion|gliden64"),
            "angrylion"
        );
    }
}
