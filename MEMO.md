# Memo: the inbox triage service

**To:** Customer-support manager · **From:** Sher Partap Singh · **14 October 2026**

For each chat, the service suggests the topic and the next step the written procedure allows, and sends the chat
to a person when unsure or off procedure. Figures come from 2,300 sample chats from a public research set (ABCD),
300 made messy on purpose. I tuned it on them, so expect lower scores on your mail.

## What it gets right and wrong

- Right next step for 84% of chats (79% of messy ones). It handles three in four alone, 91.5% of them right,
  including 103 of 111 money or account steps (refunds, order and account changes).
- About 6 in 100 chats got a wrong step with no person checking; 1 in 300 a wrong money or account step.
- Topic right 71% of the time; half the time when the customer describes a screenshot.
- A chat with two requests gets one answer; the second request alone does not send it to a person.
- All 9 junk and trick messages went to a person, but reworded planted orders are not caught.

## How much reaches a person, and why

24% of chats (34% of messy ones). I priced a wrong step taken alone at 10 staff reviews for money and account
steps and 3 for the rest. The cut-offs in use cost least at those prices: about 40 reviews per 100 chats, against
100 if a person read all. On the held-out training chats, prices of 5 and 3 send 16% to a person, and 10 and 5
send 32%.

## Cost, and your choices

1. **The AI step** (GPT-6 Luna, through OpenRouter) reads one chat in five. It lifts right steps from 81% to
   84% and saves about 120 reviews a day at 3,000 messages, for $0.125 per 1,000 messages at list price.
   Against it: customer text leaves the company, two runs differed on 71 of 2,300 chats, and the slowest 1 in 20
   answers take 1.9 s, against 38 ms without it. One setting turns it off; if it fails, our own model answers.
   Please confirm, with legal if needed, that customer text may go to it.
2. **Off-procedure steps go to a person**, at about half a review per 100 messages; this can be switched off.

Review time is the larger cost: about 730 reviews a day.

## Monday morning checks

Each figure comes from `/metrics`, saved daily (RUNBOOK.md, "Metrics"). "Normal" comes from the two live runs,
the AI-off figure from the CPU-only dev run. The limits are my guess.

| Check | Normal | Look into it if |
|---|---|---|
| Sent to a person | about 25% (about 28% with AI off) | above 35% or below 15% |
| Unreadable messages | 2 in 2,309 | above 1% |
| AI calls timed out | about 1% | above 5% |
| AI step | on | paused after failures |
| AI model named in the log | `openai/gpt-6-luna` | any other name |
| Off-procedure steps | about 4% | above 8%, or a jump after a policy edit |

Wrong but well-formed AI answers move none of these.

## The risk that remains, and when to stop

The risk is the 6 in 100 wrong steps taken with no one checking, likely more on your mail. Customers bear it
first (wrong or slow help, a second contact); the company bears refunds and account changes made in error.

Run it two weeks with a person still reading every message; go live only if at most 30 of 200 chats it would have
handled alone are wrong, and reset the limits above from the trial. Then send all work to a person if a weekly
check of 200 it handled alone finds more than 30 wrong (sample chats: about 17) or 5 or more wrong money or account
steps (about 1), or if the share sent stays above the line for a day.

<sub>Sources: `results/raw/` `dev-http-llm` (submitted), `dev-http-llm-r1`, `dev-pipeline-cpu`, `dev-http-cpu`;
EXPERIMENT_LOG P2-1, P4-1, P4-3.</sub>
