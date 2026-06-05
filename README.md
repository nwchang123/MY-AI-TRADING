# Moomoo MY Small-Cap Options Agent

Foundation for an autonomous, event-driven U.S. options experiment using a
dedicated Moomoo MY account.

The LLM does not call broker methods directly. It produces structured proposals
that must pass a deterministic risk gate before the execution adapter can place
an order.

## Current State

- OpenD environment checks
- Moomoo MY account discovery
- paper-only limit-order adapter
- immutable mandate files
- `runtime/HALT` kill switch
- proposal schema and deterministic risk gate
- append-only JSONL audit writer
- deterministic event scoring
- LLM research committee (5 roles) over an OpenAI-compatible endpoint
- U.S. small-cap screener and option-chain fetcher (Moomoo quote API)
- deterministic option liquidity validator
- SQLite + JSONL candidate snapshot store
- SEC EDGAR catalyst ingestion and news normalization
- deduplication and deterministic evidence-to-score mapping
- autonomous paper cycle: reconcile, exits, gated entries
- position monitor (take-profit, stop-loss, time stop, forced close)
- local position ledger and daily audit-log report
- offline backtest v0 for proposal/quote replay through the risk and exit engine
- gated USD 100 live run: allowlist, ramp cap, circuit-breaker HALT

## Controlled Live Run (Phase 6)

`run-live` runs one real-money cycle. It refuses to start unless every guard
passes, in order:

1. `TRADING_AGENT_MODE=live`
2. `TRADING_AGENT_ENABLE_LIVE=I_UNDERSTAND_REAL_MONEY` (explicit acknowledgment)
3. `TRADING_AGENT_ACCOUNT_ID` is set (pinned account)
4. `TRADING_AGENT_LIVE_ACCOUNT_ALLOWLIST` is set and contains the pinned account

The agent never unlocks live trading — you unlock it manually in the OpenD GUI
(a pre-flight checklist prints to stderr). REAL `TrdEnv` exists only in the
broker adapter and is reachable only through these guards. Additional live
safety:

- caps to **one** open position for the first 10 live trades (ramp), then the
  mandate maximum;
- a circuit breaker writes the `HALT` kill switch when the daily loss stop or
  hard drawdown stop is breached, stopping the session until you review and
  `resume`.

```powershell
trading-agent run-live --tickers ABCD
trading-agent report   # end-of-day review
```

Live order placement is implemented but disabled by default; it activates only
through the guards above after a completed paper shadow run.

## Autonomous Paper Cycle

`run-cycle` runs one full cycle (requires OpenD and a pinned
`TRADING_AGENT_ACCOUNT_ID`): it reconciles broker state against the local
position ledger, runs exits through the position monitor, then routes every
entry through catalysts -> committee -> liquidity validator -> deterministic
risk gate. The gate is the only path to an order, so the LLM cannot bypass the
mandate. Each ticker and position is handled once per cycle (no retry loop).

```powershell
trading-agent run-cycle --tickers ABCD,WXYZ
trading-agent positions
trading-agent report --date 2026-06-02
```

Schedule `run-cycle` on an external timer (Task Scheduler / cron) for the
20-session paper shadow run; the `HALT` kill switch aborts a cycle before any
broker call.

## Offline Backtest v0

`backtest` replays point-in-time proposals and option quotes through the same
liquidity validator, deterministic risk gate, and position monitor used by the
paper/live cycle. It does not connect to Moomoo and does not call the LLM. Buy
limits fill only when the quote ask is at or below the proposal limit; long
options are marked and exited at the bid.

```powershell
trading-agent backtest --input examples/backtest-v0.example.json
```

Use this first to test loss stops, position caps, spread costs, unfilled orders,
and exit behavior before spending API calls or running paper/live cycles.

## Public Catalyst Pipeline

`catalysts` pulls a ticker's recent SEC EDGAR filings (8-K, 10-Q/K, S-3, 424B,
13D/G, Form 4), normalizes and deduplicates them, and derives deterministic score
inputs (recent catalysts raise the score; dilution filings lower it; stale or
duplicate items do not inflate it). EDGAR requires a contactful User-Agent via
`TRADING_AGENT_SEC_USER_AGENT`.

```powershell
trading-agent catalysts --ticker NVDA --out runtime/nvda.json
```

The output bundle is directly consumable by the committee:

```powershell
trading-agent committee-run --input runtime/nvda.json
```

## Scanner And Contract Selection

`scan` screens U.S. small-cap optionable equities through OpenD using the
universe mandate (price, market cap, turnover), then keeps only names with a
usable option chain:

```powershell
trading-agent scan
```

`validate-contract` runs the deterministic liquidity validator over an option
quote (offline) and stores the snapshot in `runtime/snapshots.sqlite`. It
rejects zero bids, crossed markets, wide spreads, low OI/volume, stale quotes,
out-of-window DTE, and contracts above the cost cap:

```powershell
trading-agent validate-contract --input examples/validate-contract.approved.json
```

## LLM Research Committee

The committee (`catalyst_analyst`, `skeptic`, `options_analyst`, `risk_manager`,
`portfolio_manager`) runs against any OpenAI-compatible endpoint, using up to
three model tiers: a fast "flash" model for the idea-generating roles, an
optional cross-provider "adversary" model for the `skeptic` and `risk_manager`,
and a stronger "pro" model for the decisive `portfolio_manager`. Defaults target
DeepSeek V4; override via `.env`:

```env
TRADING_AGENT_LLM_API_KEY=sk-...
TRADING_AGENT_LLM_BASE_URL=https://api.deepseek.com
TRADING_AGENT_LLM_MODEL=deepseek-v4-flash
TRADING_AGENT_LLM_MODEL_PRO=deepseek-v4-pro
# Optional: run the veto roles on a different provider for an independent
# second opinion. Blank model = reuse the primary model.
TRADING_AGENT_LLM_ADVERSARY_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
TRADING_AGENT_LLM_ADVERSARY_API_KEY=...
TRADING_AGENT_LLM_ADVERSARY_MODEL=gemini-2.0-flash
```

Pointing the `skeptic` / `risk_manager` at a different model lineage gives their
vetoes genuinely uncorrelated errors, instead of the same model grading its own
work. Before any LLM call, a deterministic red-flag detector
(`research/redflags.py`) blocks the trade outright on a fresh dilution shelf or
insider selling (zero API cost) and hands softer warnings (earnings IV-crush,
stale catalyst, thin evidence) to the veto roles as confirmed facts.

The committee never calls the broker or the risk gate. It emits a schema-checked
proposal; invalid JSON, uncited evidence, a ticker mismatch, a deterministic
critical red flag, or a skeptic / risk-manager veto all force a `reject`. Run it
over an evidence bundle:

```powershell
trading-agent committee-run --input examples/committee-run.example.json
```

An `open_position` result can then be fed into `proposal-check` for the
deterministic risk gate.

## Setup

1. Launch and log in to the OpenD GUI.
2. Copy `.env.example` to `.env`.
3. Install the package:

   ```powershell
   python -m pip install -e .[dev]
   ```

4. Check OpenD:

   ```powershell
   trading-agent doctor
   trading-agent accounts
   ```

5. Activate or clear the local kill switch:

   ```powershell
   trading-agent halt --reason "operator requested stop"
   trading-agent resume
   ```

6. Evaluate the bundled offline example through the deterministic risk gate:

   ```powershell
   trading-agent proposal-check --input examples/proposal-check.approved.json
   ```

7. Run tests:

   ```powershell
   python -m pytest
   ```

See [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for the full delivery plan.
