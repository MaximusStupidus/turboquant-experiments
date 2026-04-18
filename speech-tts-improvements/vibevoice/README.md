# VibeVoice experiments (paused)

This folder preserves our attempt to apply TurboQuant to Microsoft's
VibeVoice-Realtime-0.5B. Work is paused due to cache-API incompatibility
between VibeVoice's internal shim (`MockCacheLayer`) and current
transformers versions.

**See `notes/part2-vibevoice-blocked.md` in the repo root for full
diagnosis and a path to revival.**

The scripts here were the 4 monkey-patches + the cache-rebuild shim we
developed. They're kept for future reference when VibeVoice gets updated
(or when someone forks it to complete the compat layer).
