# Publication privacy review

The Ring profile was assembled from selected source files, configuration templates, and synthetic validation results in a fresh clone of the public upstream repository. Private working directories, SSH controllers, credentials, real node addresses, hostnames, raw machine logs, and local account paths were excluded.

Before publication, the working tree and reachable Git history were scanned with Gitleaks 8.30.1, with report values redacted. A second check searched for deployment-specific identity/address/path patterns and unexpected opaque artifacts. Modified source and synthetic results were reviewed as well. The generated image fixture has only PNG image chunks and contains no screenshot or metadata from the operator's computer. The publication uses a GitHub noreply commit address.

Gitleaks identified tensor-call arguments involving the output variable `topk_indices` as a generic API key in copies of `sparse_attn_indexer.py`. These were individually reviewed as false positives: there is no credential string on that line. The same pre-existing false positives occur in the upstream history. No broad scanner exclusions were added for them.

No unresolved credential or private deployment-identity findings remained at publication. This records the checks performed; it is not a guarantee about every future contribution. Original already-public upstream history, attribution and example configurations are retained.
