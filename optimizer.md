## Notes:

Findings:

- Need to adjust grading: give clear rubics and instructions to increase performance. Ex: grade too hard on formating, grade on its knowledge instead of testset(which could be wrong), detail grading example for a 0, 50, 80,100 score.
- Need to select good model for grading: weaker models bad at following complex instructions.
- Model gives examples from testset lol(data leak), since I show its bad performance exs to improve. --> instruct to use similar but not exact examples.
- Model's getting good at prompt engineering

Todo:

- [ ] Test with opensource models
- [ ] Anthropic
- [ ] Deal with add in 100 score into poor performance questinos
- [ ] Human feedback

# How the AI Prompt Improvement App Works

Imagine you're trying to teach a smart robot to give better answers. This app does just that, but for AI language models. Here's how it works, step by step:

## 1. Getting Started

- The app starts with a set of questions and their correct answers.
- It also has an initial set of instructions (we call this a "prompt") for the AI.

## 2. Testing the AI

- The app asks the AI all the questions using the current instructions.
- It's like giving a quiz to the AI.

## 3. Grading the Answers

- Another, smarter AI checks how good the answers are.
- Each answer gets a score out of 100.

## 4. Calculating the Overall Score

- The app adds up all the scores and finds the average.
- This tells us how well the current instructions are working.

## 5. Trying to Improve

- The app looks at the questions the AI struggled with the most.
- It then asks another AI, "How can we change the instructions to do better on these tricky questions?"
- This AI suggests new, improved instructions.

## 6. Repeat and Compare

- The app tries these new instructions and tests the AI again.
- If the new score is better, it keeps the new instructions.
- If not, it tries to improve again.

## 7. Knowing When to Stop

- The app keeps trying to improve until one of these happens:
  1. The score is really high (95 or above out of 100).
  2. It hasn't seen any improvement for a while.
  3. It has tried a certain number of times (set by the user).

## 8. Finishing Up

- At the end, the app shows:
  - How good the original instructions were
  - How good the best new instructions are
  - How much improvement was made

Think of it like a teacher working with a student. The app tests the AI, sees where it needs help, gives it better instructions, and keeps doing this until the AI is giving great answers!
