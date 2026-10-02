# Demo script (2–3 minutes)

Spoken walkthrough for a video or a call. Short sentences. Show the Telegram bot on screen,
then the README results table. Words in *italics* are things to do, not to say.

---

**1. The problem (20 s)**

Every protocol has developer docs. People ask about them all day, in Discord and Telegram.
An AI bot sounds like the obvious answer. The risk is that it makes things up.
In DeFi, a wrong contract address or a wrong fee is not a typo. It can cost someone money.

**2. What I built (20 s)**

So I built a docs assistant for Uniswap that is honest about what it knows.
It answers from the official docs only. Every answer links to the exact section it used.
And if the docs don't cover a question, it says so instead of guessing.

**3. Live demo (60 s)**

*Open Telegram. Ask: "How many hooks can a Uniswap v4 pool have?"*

It answers: one hook per pool. And here is the link to the Hooks page it came from.

*Ask: "What was Uniswap's total trading volume in 2023?"*

The docs don't contain that. So it says it doesn't know, and points to the closest section.
No made-up number.

*Ask: "Why can each v4 pool have up to five hooks?"*

This question has a false assumption built in. The bot catches it.
It says the premise is wrong, the docs say one hook, and it cites the page.

**4. The evidence (40 s)**

*Show the results table in the README.*

I didn't just test it by hand. I built an evaluation with 80 test questions the system never
saw during development. Some are easy lookups. Some need two sections. Some are traps:
questions the docs can't answer, false premises, and questions about other protocols.

The shipped version declined every one of the 23 questions it should not answer.
Less than one percent of the claims in its answers were unsupported by the docs.
A typical "use the context" setup, on the same questions, had about five percent,
and it confidently invented an answer to a question the docs don't cover.

I'm also upfront about the limits. With 80 questions, some of these differences are not
statistically certain yet, and the README says exactly which ones are.

**5. Why it matters for your protocol (20 s)**

The same pipeline works for any protocol's docs. You change one config file and rerun.
It runs on a small server, and every answer is logged, so you can see what users really ask
and where your docs have gaps.

If you'd like this for your docs, I'm happy to set it up and show you the numbers on your own
documentation.
