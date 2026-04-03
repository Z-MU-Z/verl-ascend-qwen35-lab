# Agent Working Notes

This repository is used together with remote Huawei servers. Follow the deployment workflow below for any code or script change that is meant to run on the remote machines.

## Non-Negotiable Rule

For any change that will be used on the remote Huawei machines, the required order is:

1. Edit locally in this repository.
2. Validate locally as much as possible.
3. Commit and push from the local machine.
4. Log in to the remote machine and run `git pull`.
5. Only then run, debug, or continue work on the remote machine.

Do not skip this order. Remote machines are deployment targets, not the source of truth.

## Required Remote Sync Workflow

For any repo change that should be used on the remote servers:

1. Make the change locally in this repository first.
2. Validate the local change as much as possible.
3. Commit and push the change from the local machine to the remote git repository.
4. Log in to the remote machine and `git pull` the new change there.
5. Only after the remote `git pull` completes, continue with running or debugging on the remote machine.

## Do Not Edit Only On Remote

- Do not treat the remote machine as the source of truth.
- Do not make one-off code edits only on the remote machine unless the user explicitly asks for a temporary hotfix.
- If a temporary remote hotfix is unavoidable, mirror it back to the local repository immediately and push it so the repo stays authoritative.

## Shared Storage Context

- Shared data path: `/shared`
- Shared code shortcut: `/home/zmz/code`
- Shared training data shortcut: `/home/zmz/verl/data`
- Shared model shortcut: `/home/zmz/verl/models`

When later agents prepare code, models, or datasets for the remote machines, they should assume the local repository remains the canonical source and remote machines should sync from git after local changes are pushed.
