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
Rewrite each supplied resume bullet to be more specific, results-oriented, and
compelling. Use strong action verbs, but use only information stated or clearly
implied by the original bullet. Never invent percentages, dollar amounts,
deadlines, job duties, qualifications, achievements, or other results. If a
metric or detail is missing, do not guess it or add a placeholder.

Return only a valid JSON list with exactly one item for every supplied bullet,
in the same order as the input. Every item must contain exactly two string keys:
"original" and "improved". Copy the supplied bullet exactly into "original" and
place the rewritten version in "improved". Do not include Markdown, code fences,
an introduction, explanations, or any text before or after the JSON list.

Bullet points:
```
{bullet_text}
```
"""

    response = get_completion(
        [{"role": "user", "content": prompt}],
        temperature=0,
    )

    try:
        rewritten = json.loads(response)
        if not isinstance(rewritten, list):
            raise ValueError("The response is not a JSON list.")

        if len(rewritten) != len(bullets):
            raise ValueError("The model did not return one item per supplied bullet.")

        for original_bullet, item in zip(bullets, rewritten):
            if not isinstance(item, dict) or set(item) != {"original", "improved"}:
                raise ValueError(
                    "Every item must contain only original and improved."
                )
            if not isinstance(item["original"], str) or not isinstance(
                item["improved"], str
            ):
                raise ValueError("The original and improved values must be strings.")
            if item["original"] != original_bullet:
                raise ValueError(
                    "An original bullet was changed or returned out of order."
                )

        return rewritten
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        print("The bullet rewrite could not be parsed as valid JSON.")
        print("Error:", error)
        print("Raw response:", response)
        return []


def format_bullet_rewrites(rewritten: list[dict]) -> str:
    sections = []
    for item in rewritten:
        sections.append(
            f"Original: {item['original']}\nImproved: {item['improved']}"
        )
    return "\n\n".join(sections)


def display_bullet_rewrites(rewritten: list[dict]) -> None:
    print(format_bullet_rewrites(rewritten))


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


def add_user_to_history(messages: list[dict], user_content: str) -> None:
    messages.append({"role": "user", "content": user_content})


def add_assistant_to_history(messages: list[dict], assistant_content: str) -> None:
    messages.append({"role": "assistant", "content": assistant_content})


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
                formatted_rewrites = format_bullet_rewrites(rewritten)
                assistant_reply = (
                    f"Here are the rewritten bullet points:\n\n{formatted_rewrites}\n\n"
                    "Please review and edit these suggestions before submitting them."
                )
                print("\nJob Application Helper:")
                print(assistant_reply)
                print()

                user_turn = (
                    f"{user_input}\nHere are my bullet points:\n"
                    + "\n".join(raw_bullets)
                )
                add_user_to_history(messages, user_turn)
                add_assistant_to_history(messages, assistant_reply)

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
            assistant_reply = (
                f"{opening}\n\n"
                "Please review and edit this draft before submitting it."
            )
            print("\nJob Application Helper:")
            print(assistant_reply)
            print()

            user_turn = (
                f"{user_input}\nJob title: {job_title}\n"
                f"Background: {background}"
            )
            add_user_to_history(messages, user_turn)
            add_assistant_to_history(messages, assistant_reply)

        else:
            add_user_to_history(messages, user_input)
            reply = get_completion(messages)
            print("\nJob Application Helper:")
            print(reply)
            print()
            add_assistant_to_history(messages, reply)


if __name__ == "__main__":
    # Run this once while verifying Task 4, then comment it out if you do not want
    # the moderation tests to print every time the program starts.
    test_moderation()

    starter_bullets = [
        "Helped customers with their problems",
        "Made reports for the management team",
        "Worked with a team to finish the project on time",
    ]
    rewritten_bullets = rewrite_bullets(starter_bullets)
    print("\nTask 2 - Bullet Rewriter Test:")
    display_bullet_rewrites(rewritten_bullets)

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
# Question 1 - Bias:
# Because the chatbot learned from human-created text, it may repeat biases found
# in that data. For example, it could favor a formal corporate communication style
# and treat it as more professional than wording used in another culture or
# industry. This could make its advice less helpful or fair for some job seekers.
#
# Question 2 - Submitting output without review:
# A job seeker could submit a false or misleading application if the chatbot
# invents information. In my test, it added percentages and results that were not
# present in the original resume bullets. An employer could question the claim in
# an interview, discover that it is unsupported, and lose trust in the applicant.
# The output could also contain inaccurate role-specific advice because the model
# may not know the expectations of that company or industry.
#
# Question 3 - Professional guardrails:
# I would require users to review and approve every response before exporting it.
# I would also prevent the model from adding unsupported numbers, qualifications,
# or experiences and ask the user to provide missing details instead. Moderation,
# privacy warnings, bias testing with people from different backgrounds, and clear
# disclosure that the text was AI-assisted would further reduce possible harm.
