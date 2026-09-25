"""Gemini on Vertex AI through ChatGoogleGenerativeAI (gcloud auth application-default login; GOOGLE_CLOUD_PROJECT).

langchain-google-genai replaces the deprecated ChatVertexAI for Vertex AI.
"""

import os

from _schema import TEXTS, Ticket
from langchain_google_genai import ChatGoogleGenerativeAI

import decisionsmith as ds

project = os.environ["GOOGLE_CLOUD_PROJECT"]
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", vertexai=True, project=project, location="us-central1")
h = ds.harness(Ticket, teacher=llm, student="laya", mode="shadow")
h.many(TEXTS)
print(h.status())
