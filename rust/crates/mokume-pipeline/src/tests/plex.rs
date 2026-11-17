use super::*;

mod cases {
    use super::*;

    #[test]
    fn sample_plex_recognizes_tmt_channel_suffixes() {
        assert_eq!(sample_plex("UPS1_Norm_Mixture1_126"), "mixture1");
        assert_eq!(sample_plex("UPS1_0.5_Mixture1_127N"), "mixture1");
        assert_eq!(sample_plex("UPS1_0.5_Mixture1_127C"), "mixture1");
        assert_eq!(sample_plex("p2_127N"), "p2");
        assert_eq!(sample_plex("sample_alpha"), "plex1");
    }
}
