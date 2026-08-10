import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load API key from .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

# Create Gemini client
client = genai.Client(api_key=api_key)


# -----------------------------
# TOOL 1: Calculator
# -----------------------------

def calculator(a: float, b: float, operation: str) -> float:
    """
    Performs a mathematical operation on two numbers.

    operation can be:
    - add
    - subtract
    - multiply
    - divide
    """

    if operation == "add":
        return a + b

    elif operation == "subtract":
        return a - b

    elif operation == "multiply":
        return a * b

    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

    else:
        raise ValueError("Unknown operation")


# -----------------------------
# AGENT CONFIGURATION
# -----------------------------

config = types.GenerateContentConfig(
    system_instruction="""
You are a helpful AI agent.

You have access to a calculator tool.

Use the calculator whenever the user asks you to perform
addition, subtraction, multiplication, or division.

For normal questions, answer directly.

Explain the final answer clearly and concisely.
""",
    tools=[calculator],
    temperature=0.2
)


# -----------------------------
# AGENT FUNCTION
# -----------------------------

def ask_agent(question):

    print("\n" + "=" * 60)
    print("USER:")
    print(question)

    # First call to Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=question,
        config=config
    )

    # Check whether Gemini requested a tool
    if response.function_calls:

        print("\nAGENT: I need to use a tool.")

        function_call = response.function_calls[0]

        tool_name = function_call.name
        arguments = function_call.args

        print(f"TOOL: {tool_name}")
        print(f"ARGUMENTS: {dict(arguments)}")

        # Execute the tool
        if tool_name == "calculator":

            result = calculator(
                a=float(arguments["a"]),
                b=float(arguments["b"]),
                operation=arguments["operation"]
            )

            print(f"TOOL RESULT: {result}")

            # Send tool result back to Gemini
            final_response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=question)
                        ]
                    ),

                    response.candidates[0].content,

                    types.Content(
                        role="tool",
                        parts=[
                            types.Part.from_function_response(
                                name=tool_name,
                                response={
                                    "result": result
                                }
                            )
                        ]
                    )
                ],
                config=config
            )

            print("\nAGENT FINAL ANSWER:")
            print(final_response.text)

    else:

        print("\nAGENT FINAL ANSWER:")
        print(response.text)


# -----------------------------
# CHAT LOOP
# -----------------------------

print("🤖 Gemini AI Agent Started")
print("Type 'exit' to stop.")

while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Agent stopped.")
        break

    ask_agent(user_input)