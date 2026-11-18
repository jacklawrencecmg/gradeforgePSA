# Moving GradeForge to Claude Code

1. Install Claude Code (https://docs.claude.com/en/docs/claude-code/overview) and Python 3.11+.
2. Unzip `gradeforge_v0.5.2.zip` to a folder, e.g. `Documents\gradeforge`.
3. Open a terminal in that folder and run:
       pip install -r requirements.txt
       claude
   Claude Code reads `CLAUDE.md` automatically, so it already knows the project, the rules and the to-do list.
4. First prompts to try:
   - "Run the test suite in the background and tell me the result."
   - "Start the app and test the login, the free grade and the paywall."
   - "Walk me through connecting Stripe test mode, using PAYWALL_SETUP.md."
   - (when you have real photos) "Here are photos of a card PSA graded 9. Show which flags pulled the grade down."
5. To run with login/payments on your own machine copy `gradeforge.env.example` to `gradeforge.env`, fill it in, then start the server.

Your saved analyses and accounts live in the `data` folder, which is not part of the zip. The zip contains code only, so earlier analyses from the Windows package do not carry over unless you copy that package's `data` folder into the new project folder.
