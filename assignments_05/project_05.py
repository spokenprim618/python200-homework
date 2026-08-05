# Task 6
# The specification of speach styles is biased to certain audiences that would respond differently to certain styles.
# It may give answers the model learned that persona would give which may not be what is needed.
# The guard rails I will put is checks for more moderation resiliance in case someone tries to use this for evil purposes. 
# I would want the model output to be reviewed as well to make sure they are not giving very biased responses.
# There would need to be some checks also on the models accuracy.


from dotenv import load_dotenv
from openai import OpenAI
import json

# Task 1

load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content


system_prompt = """
You are an experienced job application coach who helps students and early-career
professionals improve resumes, cover letters, and interview responses.

Your role is to provide constructive, professional, and encouraging feedback
focused only on job application materials.

Always:
- Stay focused on resumes, cover letters, interview preparation, and related career documents.
- Suggest improvements without inventing qualifications or experiences.
- Remind the user to carefully review and edit your suggestions before submitting them anywhere.
- Acknowledge that you may not know the norms or expectations of the user's specific industry,
  company, or role, and encourage them to use their own judgment.
- Be clear, concise, and supportive.
"""

print(system_prompt)

# commnets

# Task 2

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    # Your code here: call get_completion(), parse the JSON, and return the result

    response = get_completion(messages)

    rewritten = json.loads(response)

    for item in rewritten:
        print("Original:")
        print(item["original"])

        print("Improved:")
        print(item["improved"])

    return rewritten

bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]


rewrite_bullets(bullets)

# Task 3


def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    # Your code here: call get_completion() and return the result
    return get_completion(messages)





job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed \
a Python course and built data pipelines using Prefect and Pandas."

print(generate_cover_letter(job_title, background))

# Task 4

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    # Your code here: return True if safe, False if flagged, and print a message if flagged
    if flagged:
        print("Message was flagged please try again.")
        return False

    return True


# Task 5

def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": YOUR_SYSTEM_PROMPT}
    ]

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

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            # YOUR CODE: call rewrite_bullets() and print the results
            rewritten = rewrite_bullets(raw_bullets)


            for item in rewritten:
                print("Original:")
                print(item["original"])

                print("Improved:")
                print(item["improved"])

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            # YOUR CODE: call generate_cover_letter() and print the result
            opening = generate_cover_letter(job_title, background)
            print(opening)

        # 7. Otherwise, handle it as a regular chat turn
        else:
            # YOUR CODE:
            # - Append the user's message to `messages`
            # - Call get_completion(messages)
            # - Print the reply
            # - Append the reply to `messages` as an assistant message
            messages.append({"role":"user","content":user_input})

            reply = get_completion(messages)

            print()

            print("Job Application Helper:")

            print(reply)

            print()

            messages.append({"role":"assistant","content":reply})
                        


if __name__ == "__main__":
    run_chatbot()