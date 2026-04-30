from pydantic import BaseModel, Field
from typing import List
from utils import prompts
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableMap
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser


class ModelResponse(BaseModel):
    sub_questions: List[str] = Field(description="List of sub-questions")


def subquery_chain():
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    ).with_structured_output(ModelResponse)

    return prompts.subqueries_prompt | model


def build_rag_chain(vectorstore):
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    sub_chain = subquery_chain()

    def get_context(subqueries):
        all_docs = []
        for q in subqueries.sub_questions:
            docs = retriever.invoke(q)
            all_docs.extend(docs)

        unique_docs = {doc.page_content: doc for doc in all_docs}.values()

        return "\n\n".join([doc.page_content for doc in unique_docs])

    main_chain = (
        RunnableMap({
            "subqueries": sub_chain,
            "query": RunnablePassthrough()
        })
        | {
            "context": RunnableLambda(lambda x: get_context(x["subqueries"])),
            "query": RunnableLambda(lambda x: x["query"])
        }
        | prompts.main_prompt
        | model
        | StrOutputParser()
    )

    return main_chain