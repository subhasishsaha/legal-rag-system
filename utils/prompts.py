from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

subqueries_prompt = PromptTemplate(
    template="""
    You are a legal query decomposition system.

    Given a user question, generate at most 2 concise legal sub-questions
    that are necessary to answer the query.

    Rules:
    - Maximum 2 sub-questions
    - Each must be short and precise
    - No explanations
    - No redundancy
    - Output only the questions as a numbered list

    Question: {query}
    """,
        input_variables=["query"]
)

main_prompt = ChatPromptTemplate(
    [
        ("system", 
         "You are a Legal Research Agent. Answer the user's question using the provided context. "
         "Do not list sub-questions or intermediate steps. "
         "Provide a clear, final, well-structured answer. "
         "If information is missing, say you don't know."),
        
        ("human", 
         "Question: {query}\n\nContext:\n{context}\n\n"
         "Give a final answer to the question.")
    ]
)