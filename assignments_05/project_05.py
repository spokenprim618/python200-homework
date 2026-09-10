from dotenv import load_dotenv
from openai import OpenAI
import json


# --- Task 1: Setup and System Prompt ---

load_dotenv()
client = OpenAI()


def get_completion(
    messages: list[dict], model: str = "gpt-4o-mini", temperature: float = 0.7
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400,
    )
    return response.choices[0].message.content


system_prompt = """
You are an experienced job application coach who helps students, career changers,
and early-career professionals improve resumes, cover letters, and interview
responses. Provide constructive, professional, and encouraging feedback focused
only on job application materials.

Always:
- Stay focused on resumes, cover letters, interviews, and related career documents.
- Improve the user's wording without inventing numbers, qualifications, or experiences.
- Remind the user to carefully review and edit every suggestion before submitting it.
- Acknowledge that you may not know the norms of the user's specific industry,
  company, or role, and encourage the user to apply their own judgment.
- Be clear, concise, and supportive.
"""

# I explicitly prohibited invented numbers and qualifications because unsupported
# claims could cause a user to submit misleading information to an employer.


# --- Task 2: Bullet Point Rewriter ---

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    bullet_text = "\n".join(f"- {bullet}" for bullet in bullets)

    prompt = f"""
You are a professional resume coach helping a career changer.
Rewrite each resume bullet to be clearer, more specific, and results-oriented.
Use strong action verbs, but do not invent numbers, results, duties, or facts.
If a useful detail is missing, improve only the wording that is supported.

Return only a valid JSON list. Every item must have exactly two string keys:
"original" and "improved". Include one item for every supplied bullet. Do not
include Markdown, code fences, an introduction, or text after the JSON.

Bullet points:
```
{bullet_text}
```
"""

    response = get_completion([{"role": "user", "content": prompt}])

    try:
        rewritten = json.loads(response)
        if not isinstance(rewritten, list):
            raise ValueError("The response is not a JSON list.")

        for item in rewritten:
            if not isinstance(item, dict) or not {
                "original",
                "improved",
            }.issubset(item):
                raise ValueError("A result is missing original or improved.")

            print("\nOriginal:", item["original"])
            print("Improved:", item["improved"])

        return rewritten
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        print("The bullet rewrite could not be parsed as valid JSON.")
        print("Error:", error)
        print("Raw response:", response)
        return []


# The starter bullets are weak because they use vague verbs and do not explain
# the work clearly. The model should suggest stronger verbs and clearer wording,
# but it must not invent percentages, deadlines, or outcomes.


# --- Task 3: Cover Letter Generator ---

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
You write strong cover letter opening paragraphs for career changers.
Write 3-5 sentences that are confident, specific, and free of clichés.
Use only facts stated in the person's background. Do not invent achievements.

Example 1:
Role: Data Analyst at a healthcare nonprofit
Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
Opening: After seven years as a registered nurse, I've spent my career making decisions
under pressure using incomplete information, which is excellent preparation for data
analysis. I recently completed a data analytics program where I built dashboards
tracking patient outcomes across departments. I'm excited to bring that combination
of clinical context and technical skill to [Company]'s mission-driven work.

Example 2:
Role: Junior Software Engineer at a fintech startup
Background: Ten years in retail banking operations, self-taught Python developer for two years.
Opening: I spent a decade on the operations side of banking, watching technology
decisions affect processes I understood firsthand. That experience led me to study
Python for two years and prepare for work in software engineering. I'm applying to
[Company] because its payment infrastructure work connects my banking experience
with my new technical skills.

Now write an opening paragraph for this person:
Role: {job_title}
Background: {background}
Opening:
"""

    return get_completion([{"role": "user", "content": prompt}])


# I chose examples with two different career transitions so the model learns the
# pattern rather than copying one profession. Few-shot prompting helps control the
# paragraph's length, tone, specificity, and connection between old and new skills.


# --- Task 4: Moderation Check ---

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text,
    )
    flagged = result.results[0].flagged

    if flagged:
        print(
            "Job Application Helper: I cannot help with that wording because it "
            "was flagged by the safety check. Please rephrase it respectfully."
        )
        return False

    return True


def test_moderation() -> None:
    safe_text = "Please help me improve my data analyst resume."
    flagged_text = "I want to kill someone."

    print("Task 4 - Safe input result:", is_safe(safe_text))
    print("Task 4 - Flagged input result:", is_safe(flagged_text))


# --- Task 5: Chatbot Loop ---

def run_chatbot():
    messages = [{"role": "system", "content": system_prompt}]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        if not user_input:
            continue

        if not is_safe(user_input):
            continue

        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []

            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line and is_safe(line):
                    raw_bullets.append(line)

            if not raw_bullets:
                print("Job Application Helper: No bullet points were provided.")
                continue

            rewritten = rewrite_bullets(raw_bullets)
            if rewritten:
                print(
                    "\nJob Application Helper: Please review and edit these "
                    "suggestions before submitting them.\n"
                )
                assistant_result = json.dumps(rewritten, indent=2)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"{user_input}\nHere are my bullet points:\n"
                            + "\n".join(raw_bullets)
                        ),
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "Here are the rewritten bullet points:\n"
                            f"{assistant_result}\nPlease review and edit them before submitting."
                        ),
                    }
                )

        elif "cover letter" in user_input.lower():
            job_title = input(
                "Job Application Helper: What is the job title? "
            ).strip()
            background = input(
                "Job Application Helper: Briefly describe your background: "
            ).strip()

            if not job_title or not background:
                print("Job Application Helper: Please provide both requested details.")
                continue

            if not is_safe(job_title) or not is_safe(background):
                continue

            opening = generate_cover_letter(job_title, background)
            print("\nJob Application Helper:")
            print(opening)
            print("Please review and edit this draft before submitting it.\n")

            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"{user_input}\nJob title: {job_title}\n"
                        f"Background: {background}"
                    ),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        f"{opening}\nPlease review and edit this draft before submitting it."
                    ),
                }
            )

        else:
            messages.append({"role": "user", "content": user_input})
            reply = get_completion(messages)
            print("\nJob Application Helper:")
            print(reply)
            print()
            messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    # Run this once while verifying Task 4, then comment it out if you do not want
    # the moderation tests to print every time the program starts.
    test_moderation()

    starter_bullets = [
        "Helped customers with their problems",
        "Made reports for the management team",
        "Worked with a team to finish the project on time",
    ]
    rewrite_bullets(starter_bullets)

    test_job_title = "Junior Data Engineer"
    test_background = (
        "Five years of experience as a middle school math teacher; recently "
        "completed a Python course and built data pipelines using Prefect and Pandas."
    )
    print("\nTask 3 - Cover Letter Test:")
    print(generate_cover_letter(test_job_title, test_background))

    run_chatbot()


# --- Task 6: Ethics Reflection ---
# Chosen format: Option A - Comment block
#
# The chatbot stayed professional and focused on job applications, but the resume
# rewrites exposed an important reliability problem. It invented percentages and
# results that did not appear in the original bullets, which could cause a user to
# submit false or misleading information to an employer. Its advice may also favor
# communication styles or industries that were better represented in its training
# data. If I deployed this professionally, I would prevent unsupported numbers or
# qualifications from being added and ask the user for missing details instead. I
# would also use moderation, test the chatbot with users from different backgrounds,
# evaluate its accuracy, and require users to review every response before submitting it.
