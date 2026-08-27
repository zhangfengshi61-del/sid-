# Anonymization record

The reviewer ZIP is built from an explicit whitelist and excludes Git history.
The public source and data were audited for:

- author names, usernames, home directories, and absolute machine paths;
- cluster, container, node, job, session, and lease identifiers;
- access tokens, API keys, credentials, high-entropy secrets, and private URLs;
- raw logs, launch commands tied to private infrastructure, checkpoints, and
  cache locations;
- internal run tags in the reviewer-facing data files.

Experiment cells use descriptive public IDs such as
`K10_B40_Sci_seed42`. The ZIP builder reruns the same text scan before writing
the archive. Upstream repository names and public dataset names are retained
because they are required for reproducibility and do not identify the authors
of this submission.
