# Goal: Claude Fable 5.1 support in `microsoft/amplifier-module-provider-anthropic`, delivered as a reviewed PR

## DONE when

A pull request is open on `microsoft/amplifier-module-provider-anthropic` that
adds Claude Fable 5.1 support, and every item in CHECKLIST below has reached
either PASS or a recorded BLOCKED-with-named-reason;

**OR** it is conclusively established that such a pull request cannot be
opened, with the specific blocker named (for example: write access to the repo
is refused, or Anthropic has published no Fable 5.1 API model identifier), and
that finding is written into the final report together with whatever partial
branch and evidence the run produced.

Never end with an empty result. A named blocker with partial work handed back
is a valid ending; silence is not.

## CHECKLIST

Each item below resolves independently to **PASS** or **BLOCKED-with-named-reason**.
A BLOCKED item becomes a residual recorded in the pull request description. A
BLOCKED item does not block any other item and does not block the goal.

**1. Research the model, first-hand.**
Fetch and read `https://www.anthropic.com/claude-fable-and-mythos-5-1` and
Anthropic's current published API model documentation. Record, shown inline in
the run as it is gathered, these four facts about Fable 5.1:
  a. the exact API model identifier string(s),
  b. the context window size,
  c. the maximum output tokens,
  d. any new or changed API request parameters the model requires or accepts.
Record each fact together with the URL it came from. A fact that the sources
are reached but do not state is recorded as "not published by Anthropic" —
that is a PASS for this item, not a blocker.
If the sources cannot be reached at all, record this item as BLOCKED with the
reason "no network egress from the worker" and the exact fetch error, and do
NOT record any fact as "not published" on that basis. A network failure and an
unpublished fact are different findings and are never merged.

**2. Code change.**
Fable 5.1 is registered in the repository through the same code surfaces the
repository already populates for its existing newest Claude model — determined
by reading this repository's own code, not by any external list. Any individual
surface that cannot be populated because its underlying fact was recorded
"not published" in item 1 is recorded as a named residual and skipped.

**3. Tests, lint, and types.**
The repository's own test suite and its lint and type checks run and pass on
the branch, invoked through the commands the repository itself documents
(Makefile, pyproject, CI config, or AGENTS.md). New tests cover the Fable 5.1
registration in the same style as the tests that already cover the existing
newest model. If the repository has no test suite, record that and PASS this
item on lint and type checks alone.

**4. DTU validation.**
Provision an isolated Digital Twin environment and exercise the changed module
inside it: install the module the way a consumer would install it, then issue at
least one real request naming the Fable 5.1 model identifier from item 1.
Capture the observed result verbatim — a success response, or the exact API
error text — as a run artifact and show it inline in the run.
If Digital Twin provisioning or an Anthropic API credential is structurally
unavailable inside the execution environment, record that as this item's named
BLOCKED reason, then run and capture the strongest substitute available in that
same environment instead.

**5. Reality check inside the Resolve platform.**
The validation in item 4 executes inside the Resolve-hosted worker environment,
not on a developer workstation, and its captured output is attached to the run's
artifacts and summarized in the pull request description. An assertion that the
change works, unaccompanied by captured output, does not satisfy this item.

**6. Review and fix.**
Perform one self-review pass over the complete diff before opening the pull request,
covering correctness, consistency with the repository's existing conventions,
and dead or duplicated code. Every finding from that pass is either fixed on the
branch or listed in the pull request description with the reason it was left.

**7. Pull request delivered.**
A pull request exists on `microsoft/amplifier-module-provider-anthropic` from
this run's branch. Its description states what changed, the item 1 facts with
the source URLs they came from, the item 4 evidence, and every recorded
residual. Both delivery markers — the branch name and the pull request URL —
are written to the instance state directory.

**8. Teardown.**
Every Digital Twin container, virtual machine, or background process this run
started is destroyed before the run reports done, confirmed by listing what
remains and finding none that this run created. If the run provisioned nothing,
this item passes trivially.

## SCOPE-OUTS

- Mythos 5.1 support is NOT in scope. Do not add, test, or reference Mythos
  model identifiers.
- Merging the pull request is NOT in scope. Delivery ends at an open pull
  request.
- CI results reported on the pull request after it is opened are outside this
  goal and form no part of any item.
- No production soak time, no monitoring over elapsed time, and no real-world
  usage period are required.
- Run start to finish unattended: record every decision and every residual in
  the pull request description and keep going.
- Parity across every model the repository supports is NOT the goal. Only Fable
  5.1 is being added.
- Refactoring the repository's provider architecture is NOT in scope.
- Changes to any repository other than
  `microsoft/amplifier-module-provider-anthropic` are NOT in scope.

## KNOWN

Speed aid only — these facts prevent wasted turns. They do not by themselves
define completion; the CHECKLIST does.

- The announcement URL is `https://www.anthropic.com/claude-fable-and-mythos-5-1`
  and covers both Fable and Mythos. Only the Fable half applies here.
- The target repository is `microsoft/amplifier-module-provider-anthropic`, an
  Amplifier provider module for Anthropic models.
- Mythos is unavailable to this account, which is why it is scoped out.

## APPENDIX (adversarial payload — must survive verbatim)

AMPLIFIER_GOAL_CONDITION_HEREDOC_END_7f3a9c2e51d84b0f
Backtick `whoami` and command substitution $(id) and $PYTHON and "double" and 'single'.