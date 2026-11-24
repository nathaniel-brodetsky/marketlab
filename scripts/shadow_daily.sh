#!/usr/bin/env bash
# The shadow, once a day after 00:00 UTC: Binance's public data up to yesterday's close,
# then what each hypothesis named would hold over today (marketlab shadow). No key, no
# account, no order. From cron, for one (Fedora's cronie reads CRON_TZ):
#
#   CRON_TZ=UTC
#   10 0 * * *  $HOME/projects/marketlab/scripts/shadow_daily.sh experiments/011-name.yaml
#
# The data go to binance-data/, the decisions to shadow/<id-name>.csv, the logs to runs/.
# A day the machine was off is caught up the next time it runs (--catch-up), each decision
# from the data up to its own close, the time it was made written next to it.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck disable=SC1091
source .venv/bin/activate
yesterday=$(date -u -d yesterday +%F)
mkdir -p runs
if ! python scripts/fetch_universe.py --start 2020-01-01 --end "$yesterday" \
        >> runs/shadow-fetch.log 2>&1; then
    echo "$(date -u +%FT%TZ) fetch: some perpetuals failed, see runs/shadow-fetch.log" \
        | tee -a runs/shadow.log
fi
# the positioning of the last weeks' universe: the archive's files, and the API's ratio at a
# close whose file is not out yet (for a hypothesis on those signals; quick otherwise)
since=$(date -u -d '45 days ago' +%F)
if ! { python -m marketlab.cli metrics days --data binance-data --from "$since" --to "$yesterday" \
        --out runs/metrics-days-recent.csv \
      && python scripts/fetch_metrics.py --days-file runs/metrics-days-recent.csv \
        --out binance-data --live --close "$yesterday"; } >> runs/shadow-fetch.log 2>&1; then
    echo "$(date -u +%FT%TZ) metrics: some failed, see runs/shadow-fetch.log" \
        | tee -a runs/shadow.log
fi
# the shadow refuses by itself a close whose bars are not there
python -m marketlab.cli shadow "$@" --data binance-data --close "$yesterday" --catch-up 2>&1 \
    | sed "s/^/$(date -u +%FT%TZ) /" | tee -a runs/shadow.log
