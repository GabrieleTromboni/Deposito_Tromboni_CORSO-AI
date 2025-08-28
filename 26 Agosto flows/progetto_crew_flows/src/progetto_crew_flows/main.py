'''
Flow schema:

    Inputs: query from user about search topic, complex answer with topic inside.
    First step: scan query and extract the search topic from it.
    Second step: validate the search topic, is it into the allowed topic list? Topic list predefined.
    Third step: if valid, proceed with a RAG (Retrieval-Augmented Generation) process.
        The topic will be used to retrieve relevant documents and generate a response.
        RAG as med_crew.py (rielaborate prompt, search results into its vector database, generate answer as GuideOutline class).
    Fourth step: if not valid, start a search_crew to find relevant informations on the web about that topic (use search_crew.py)
    Fifth step: return the final response to the user and save it using OutputGuideline class independently if coming from RAG or search_crew.
    Final steps: kickoff() and plot()

Instructions for use:
    Generate documents into vector database using an LLM (Large Language Model) of different topics following what is done in rag_faiss_lmstudio.py
    and save the full topics list to know which topics are available in order to validate the search topic and retrieve relevant documents with RAG.

    First and second steps must be done with an LLM (Large Language Model) with care and precision.
    '''

