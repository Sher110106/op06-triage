# Memo: the inbox triage service

**To:** Customer-support manager · **From:** Sher Partap Singh · **14 October 2026**

For each chat, the service suggests the topic and the next step the written procedure allows, and flags the chat
for a person when unsure or off procedure. Figures come from 2,300 test chats from a public research set (ABCD),
300 made messy on purpose. I tuned it on these chats, so expect lower scores on your mail.

## What it gets right and wrong

- Right next step for 84% of messages (79% of messy ones). It handles three in four alone, 91.5% of them right,
  including 103 of 111 money or account steps (refunds, order and account changes, and the like).
- About 6 in 100 messages got a wrong step with no person checking; about 1 in 300 a wrong money or account step.
- Topic right 71% of the time, often confusing close topics; half the time when a screenshot is described.
- A message with two requests gets one answer. It noticed the second in 19 of 50 tests, but that alone does not
  send the chat to a person.
- Nine junk and trick messages all went to a person. Planted orders in known wording ("ignore your instructions")
  always do; reworded ones are not caught, and planted text changed the topic or step in one chat in four or five.

## How much reaches a person, and why

24% of messages (34% of messy ones). I priced a wrong step taken alone as 10 staff reviews for money and account
steps, 3 for the rest; the cut-offs in use cost least at those prices, about 40 reviews per 100 messages against
100 if a person read all. On 3,966 held-back chats they send 21% (92% right when not sent). At 5 and 3 reviews
they would send 16%; at 10 and 5, 32%.

## Your choices

1. **The AI step** (OpenAI's GPT-6 Luna, through OpenRouter) reads one message in five. It lifts right steps from
   81% to 84% and saves about 120 reviews a day at 3,000 messages, for about $0.38 a day ($0.125 per 1,000 at list
   price). Against it: customer text leaves the company, two runs differed on 71 of 2,300 messages, and the
   slowest 1 in 20 answers take 1.9 s or more, not 38 ms. One setting turns it off; if the AI fails, our model
   answers alone. Please confirm, with legal if needed, that customer text may go to it.
2. **Off-procedure steps to a person:** on now, at about half a review per 100 messages; can be switched off.

Half the answers take under 9 ms. Review time is the larger cost: about 730 reviews a day.

## Monday morning checks

Whoever runs the service saves `/metrics` daily and works out each share (RUNBOOK.md, "Metrics"). "Normal" is
from our two live runs; the last column is my guess.

| Check | Normal | Look into it if |
|---|---|---|
| Share sent to a person | 24% (28% with AI off) | above 35% or below 15% |
| Unreadable messages refused | 2 in 2,309 | above 1% |
| AI calls timed out | about 1% | above 5% |
| AI step | on | paused after failures |
| Off-procedure steps | about 4% | above 8%, or a jump after a policy edit |

Wrong but well-formed AI answers move none of these numbers.

## The risk that remains, and when to stop

Wrong steps taken with no one checking: 6 in 100 messages on test chats, likely more on yours. Customers bear it
first (wrong or slow help, a second contact); the company bears refunds and account changes made in error.

Run it two weeks with a person still reading every message; go live only if at most 30 of 200 it would have
handled alone are wrong, and reset the 35% and 15% lines from the trial. Then send all work to a person if a
weekly check of 200 it handled alone finds more than 30 wrong (tests: about 17; at that rate, under 1 week in
1,000) or 5 or more wrong money or account steps (tests: about 1), or the share sent stays above the line a day.

**Next quarter:** count how often the AI agrees with our own model, so a changed AI shows on Monday; label 2,000
of your messages to reset the cut-offs (staff time not yet estimated); test an in-house model that scored about
the same as the AI step, with no outside calls, but just missed the bar I set in advance.

<sub>Sources: `results/raw/dev-http-llm` (submitted), `-r1`, `dev-pipeline-cpu`, `dev-http-cpu`; `experiments/`
`routing`, `chaos`, `encoder`; EXPERIMENT_LOG P2-1, P4-1.</sub>
