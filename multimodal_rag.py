import base64
import math
import mimetypes
import re
import subprocess
import wave
from pathlib import Path

import cv2
import faiss
import imageio_ffmpeg
import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

PDF_PATH = DATA_DIR / "genai_engineer_roadmap.pdf"
IMAGE_PATH = DATA_DIR / "rag_architecture.png"
VIDEO_PATH = DATA_DIR / "rag_chunking_demo.mp4"
FRAMES_DIR = DATA_DIR / "extracted_frames"
AUDIO_DIR = DATA_DIR / "extracted_audio"
AUDIO_PATH = AUDIO_DIR / f"{VIDEO_PATH.stem}.wav"

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


def load_and_split_pdf():
    """Extract text from the PDF and split it into chunks."""

    # Load PDF and extract text page by page
    reader = PdfReader(str(PDF_PATH))
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        if page_text.strip():
            documents.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "source": PDF_PATH.name,
                        "page": page_number,
                        "modality": "pdf"
                    }
                )
            )

    # Split extracted text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    print(f"PDF pages: {len(reader.pages)}")
    print(f"Pages with extracted text: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")

    if chunks:
        print("\nSample chunk:")
        print(chunks[0].page_content)

        print("\nSample metadata:")
        print(chunks[0].metadata)

    return chunks


def process_image():
    """Generate a textual description of the RAG architecture image."""

    # Verify that the image exists and contains data
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(f"Image file not found: {IMAGE_PATH}")

    with open(IMAGE_PATH, "rb") as image_file:
        image_bytes = image_file.read()

    if not image_bytes:
        raise ValueError(f"Image file is empty: {IMAGE_PATH}")

    # Identify the image format
    mime_type, _ = mimetypes.guess_type(str(IMAGE_PATH))

    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"Unsupported image format: {IMAGE_PATH}")

    # Encode the image for Gemini
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # Prepare the prompt
    prompt = """
    Analyze this RAG architecture diagram.

    Explain:

    1. The main components shown in the diagram.
    2. The indexing process.
    3. The retrieval and generation process.
    4. How the components work together.

    Describe only the information visible in the diagram.
    Keep the explanation clear and structured.
    """

    # Send the image and prompt to Gemini Vision
    response = llm.invoke(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "base64": image_base64,
                        "mime_type": mime_type
                    }
                ]
            }
        ]
    )

    image_description = response.text

    if not image_description or not image_description.strip():
        raise RuntimeError("Gemini returned an empty image description.")

    image_description = image_description.strip()

    # Convert the description into a LangChain Document
    image_document = Document(
        page_content=image_description,
        metadata={
            "source": IMAGE_PATH.name,
            "modality": "image",
            "image_path": str(IMAGE_PATH)
        }
    )

    print("\nImage processing completed.")
    print("\nGenerated image description:")
    print(image_description)

    print("\nImage metadata:")
    print(image_document.metadata)

    return image_document


def extract_video_frames(interval_seconds=1):
    """Extract representative frames from a video at fixed time intervals."""

    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than zero.")

    # Verify that the video exists
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Video file not found: {VIDEO_PATH}")

    # Open the video file
    video = cv2.VideoCapture(str(VIDEO_PATH))

    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    extracted_frames = []

    try:
        # Read video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

        if fps <= 0 or total_frames <= 0:
            raise ValueError("Could not read valid video properties.")

        duration = total_frames / fps

        # Create the output directory
        FRAMES_DIR.mkdir(parents=True, exist_ok=True)

        # Extract one frame at each specified time interval
        timestamp = 0.0

        while timestamp < duration:
            timestamp_ms = int(round(timestamp * 1000))

            # Move to the requested timestamp
            video.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)

            success, frame = video.read()

            if not success:
                break

            # Save the extracted frame
            frame_path = FRAMES_DIR / f"frame_{timestamp_ms:06d}.jpg"

            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"Could not save frame: {frame_path}")

            extracted_frames.append(
                {
                    "source": VIDEO_PATH.name,
                    "timestamp_seconds": round(timestamp, 2),
                    "frame_path": str(frame_path)
                }
            )

            timestamp += interval_seconds

    finally:
        # Release the video resource
        video.release()

    if not extracted_frames:
        raise RuntimeError("No frames could be extracted from the video.")

    print("\nVideo frame extraction completed.")
    print(f"Video duration: {duration:.2f} seconds")
    print(f"Frames extracted: {len(extracted_frames)}")

    print("\nExtracted frame details:")

    for frame in extracted_frames:
        print(
            f"Timestamp: {frame['timestamp_seconds']}s | "
            f"File: {Path(frame['frame_path']).name}"
        )

    return extracted_frames


def describe_video_frames(extracted_frames):
    """Generate textual descriptions for the extracted video frames."""

    video_frame_descriptions = []

    for frame in extracted_frames:
        frame_path = Path(frame["frame_path"])

        if not frame_path.is_file():
            raise FileNotFoundError(
                f"Extracted frame not found: {frame_path}"
            )

        # Read and encode the frame
        with open(frame_path, "rb") as image_file:
            frame_bytes = image_file.read()

        if not frame_bytes:
            raise ValueError(f"Extracted frame is empty: {frame_path}")

        image_base64 = base64.b64encode(frame_bytes).decode("utf-8")

        # Identify the image format
        mime_type, _ = mimetypes.guess_type(str(frame_path))

        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError(
                f"Could not identify image format: {frame_path}"
            )

        # Prepare instructions for Gemini Vision
        prompt = """
        Describe the visual content of this video frame accurately.

        Follow these instructions:

        1. Describe only the content visible in the frame.
        2. Identify important text, headings, diagrams, and visual elements.
        3. Explain the main action or concept being demonstrated.
        4. Do not assume information that is not visible.
        5. Do not refer to events outside this frame.
        6. Keep the description clear, concise, and useful for
           searching and retrieving information.
        """

        # Send the frame and prompt to Gemini Vision
        response = llm.invoke(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "base64": image_base64,
                            "mime_type": mime_type
                        }
                    ]
                }
            ]
        )

        description = response.text

        if not description or not description.strip():
            raise RuntimeError(
                f"Gemini returned an empty description for: {frame_path.name}"
            )

        video_frame_descriptions.append(
            {
                "source": frame["source"],
                "timestamp_seconds": frame["timestamp_seconds"],
                "frame_path": str(frame_path),
                "description": description.strip()
            }
        )

        print(
            f"\nFrame processed: {frame_path.name} "
            f"({frame['timestamp_seconds']}s)"
        )
        print(f"Description: {description.strip()}")

    if not video_frame_descriptions:
        raise RuntimeError(
            "No video frame descriptions were generated."
        )

    print("\nVideo frame description completed.")
    print(f"Total frames described: {len(video_frame_descriptions)}")

    return video_frame_descriptions

def extract_video_audio():
    """Extract and validate the audio track from the source video."""

    # Verify that the source video exists
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Video file not found: {VIDEO_PATH}")

    # Create the output directory
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Locate the FFmpeg executable
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # Extract audio and convert it to a transcription-friendly WAV format
    command = [
        ffmpeg_exe,
        "-y",
        "-i", str(VIDEO_PATH),
        "-map", "0:a:0",
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(AUDIO_PATH)
    ]

    # Execute FFmpeg and capture any errors
    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Audio extraction failed:\n{result.stderr}"
        )

    # Verify that the extracted audio file exists
    if not AUDIO_PATH.is_file() or AUDIO_PATH.stat().st_size <= 44:
        raise RuntimeError("Extracted audio file is missing or empty.")

    # Validate the WAV file properties
    with wave.open(str(AUDIO_PATH), "rb") as audio:
        channels = audio.getnchannels()
        sample_rate = audio.getframerate()
        sample_width = audio.getsampwidth()
        frame_count = audio.getnframes()
        duration = frame_count / sample_rate

    if frame_count <= 0 or duration <= 0:
        raise ValueError("The extracted audio contains no valid samples.")

    if channels != 1 or sample_rate != 16000 or sample_width != 2:
        raise ValueError("The extracted audio does not match the expected format.")

    print("\nAudio extraction completed.")
    print(f"Audio file: {AUDIO_PATH.name}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Channels: {channels}")
    print(f"Sample width: {sample_width * 8}-bit")
    print(f"File size: {AUDIO_PATH.stat().st_size / 1024:.2f} KB")

    return AUDIO_PATH


def transcribe_video_audio():
    """Transcribe speech from the extracted video audio using Gemini."""

    # Verify that the extracted audio file exists
    if not AUDIO_PATH.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {AUDIO_PATH}. "
            "Run audio extraction first."
        )

    # Read and encode the audio file
    with open(AUDIO_PATH, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")

    # Prepare transcription instructions
    prompt = """
    Transcribe the speech in the provided audio accurately.

    Follow these instructions:

    1. Include only words that are actually spoken.
    2. Do not summarize, paraphrase, or add explanations.
    3. Preserve the original language of the speech.
    4. Divide the transcript into meaningful speech segments.
    5. Provide start and end timestamps for each segment only
       when they can be identified reliably.
    6. If a timestamp cannot be identified reliably, write "unknown".
    7. Do not invent missing words or timestamps.
    8. If no speech is audible, clearly state that no speech was detected.

    Use this output format:

    Segment 1
    Start: 00:00
    End: 00:03
    Text: [Spoken words]

    Segment 2
    Start: 00:03
    End: 00:06
    Text: [Spoken words]

    Continue for the remaining speech.
    """

    # Send the audio and transcription instructions to Gemini
    response = llm.invoke(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "media",
                        "data": audio_base64,
                        "mime_type": "audio/wav"
                    }
                ]
            }
        ]
    )

    transcript = response.text

    if not transcript or not transcript.strip():
        raise RuntimeError("Gemini returned an empty transcription.")

    print("\nAudio transcription completed.")
    print("\nGenerated transcript:")
    print(transcript)

    return transcript


def parse_transcript_segments(transcript):
    """Parse the timestamped transcript into individual segments."""

    pattern = (
        r"Segment\s+\d+\s*\n\s*"
        r"Start:\s*(.*?)\s*\n\s*"
        r"End:\s*(.*?)\s*\n\s*"
        r"Text:\s*(.*?)(?=\n\s*Segment\s+\d+\s*\n|\Z)"
    )

    transcript = transcript.strip()

    # A video may contain no audible speech. Its visual content can
    # still be indexed even when there are no transcript segments.
    if re.search(r"no speech (?:was )?detected", transcript, re.IGNORECASE):
        return []

    matches = list(
        re.finditer(
            pattern,
            transcript,
            flags=re.DOTALL | re.IGNORECASE
        )
    )

    if not matches:
        raise ValueError(
            "No transcript segments found. Check the transcript format."
        )

    transcript_segments = []

    for match in matches:
        start_time = match.group(1).strip()
        end_time = match.group(2).strip()
        segment_text = match.group(3).strip()

        if not segment_text:
            segment_number = match.group(0).splitlines()[0]
            raise ValueError(f"{segment_number} contains no transcript text.")

        transcript_segments.append({
            "start_time": start_time,
            "end_time": end_time,
            "text": segment_text
        })

    return transcript_segments


def prepare_multimodal_documents(
    pdf_chunks,
    image_document,
    video_frame_descriptions,
    transcript_segments
):
    """Combine all modality-specific representations into LangChain Documents."""

    multimodal_documents = []

    # 1. Reuse existing PDF chunks
    multimodal_documents.extend(pdf_chunks)

    # 2. Reuse the existing image description
    multimodal_documents.append(image_document)

    # 3. Convert video frame descriptions into Documents
    for frame in video_frame_descriptions:
        if not frame["description"].strip():
            raise ValueError(
                f"Empty description for frame: {frame['frame_path']}"
            )

        frame_document = Document(
            page_content=frame["description"],
            metadata={
                "source": frame["source"],
                "timestamp_seconds": frame["timestamp_seconds"],
                "frame_path": frame["frame_path"],
                "modality": "video"
            }
        )

        multimodal_documents.append(frame_document)

    # 4. Convert transcript segments into Documents
    for segment in transcript_segments:
        if not segment["text"].strip():
            raise ValueError("Transcript contains an empty segment.")

        transcript_document = Document(
            page_content=segment["text"],
            metadata={
                "source": VIDEO_PATH.name,
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "modality": "video_audio"
            }
        )

        multimodal_documents.append(transcript_document)

    # 5. Validate the combined collection
    if not multimodal_documents:
        raise ValueError("No documents available for multimodal processing.")

    if any(
        not document.page_content.strip()
        for document in multimodal_documents
    ):
        raise ValueError("The collection contains an empty document.")

    return multimodal_documents


def generate_embeddings(multimodal_documents):
    """Generate embeddings for all multimodal document representations."""

    if not multimodal_documents:
        raise ValueError("No documents available for embedding generation.")

    # Load the existing embedding model
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Extract textual content from all documents
    document_texts = [
        document.page_content
        for document in multimodal_documents
    ]

    # Generate embeddings for all document texts
    vectors = embeddings_model.embed_documents(document_texts)

    # Validate the generated embeddings
    if len(vectors) != len(multimodal_documents):
        raise RuntimeError(
            "The number of embeddings does not match the number of documents."
        )

    if not vectors or not vectors[0]:
        raise RuntimeError("Embedding generation returned empty vectors.")

    embedding_dimension = len(vectors[0])

    if any(
        len(vector) != embedding_dimension
        for vector in vectors
    ):
        raise RuntimeError("Inconsistent embedding dimensions detected.")

    if any(
        not math.isfinite(value)
        for vector in vectors
        for value in vector
    ):
        raise RuntimeError("Non-finite values detected in embeddings.")

    return embeddings_model, vectors
def store_embeddings_in_faiss(vectors, multimodal_documents):
    """Store multimodal document embeddings in a FAISS index."""

    if len(vectors) != len(multimodal_documents):
        raise ValueError(
            "The number of embeddings must match the number of documents."
        )

    # Convert embeddings into a NumPy float32 array
    vectors_array = np.array(vectors, dtype=np.float32)

    # Normalize vectors for cosine similarity search
    faiss.normalize_L2(vectors_array)

    # Create a FAISS index using inner product similarity
    embedding_dimension = vectors_array.shape[1]
    index = faiss.IndexFlatIP(embedding_dimension)

    # Add normalized embeddings to the FAISS index
    index.add(vectors_array)

    # Maintain mapping between FAISS positions and original documents
    document_mapping = {
        index_position: document
        for index_position, document in enumerate(multimodal_documents)
    }

    return index, document_mapping


def retrieve_documents(
    query,
    embeddings_model,
    faiss_index,
    document_mapping,
    top_k=5
):
    """Retrieve the most relevant documents from the FAISS index."""

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if faiss_index.ntotal == 0:
        raise ValueError("FAISS index is empty.")

    # Generate the query embedding
    query_vector = embeddings_model.embed_query(query)

    # Convert the query embedding into a NumPy float32 array
    query_array = np.array([query_vector], dtype=np.float32)

    # Normalize the query vector
    faiss.normalize_L2(query_array)

    # Search the FAISS index
    scores, indices = faiss_index.search(
        query_array,
        min(top_k, faiss_index.ntotal)
    )

    # Retrieve the original documents and their similarity scores
    retrieved_documents = []

    for score, index_position in zip(scores[0], indices[0]):
        if index_position == -1:
            continue

        document = document_mapping.get(int(index_position))

        if document is None:
            raise KeyError(
                f"No document found for FAISS index position {index_position}."
            )

        retrieved_documents.append({
            "document": document,
            "score": float(score)
        })

    return retrieved_documents


def prepare_multimodal_context(retrieved_documents):
    """Prepare retrieved text and original visual content for Gemini Vision."""

    if not retrieved_documents:
        raise ValueError("No retrieved documents available.")

    textual_context = []
    visual_context = []

    for rank, result in enumerate(retrieved_documents, start=1):
        document = result["document"]
        score = result["score"]
        metadata = document.metadata
        modality = metadata.get("modality")

        # Prepare the retrieved text and its metadata
        text_entry = {
            "rank": rank,
            "content": document.page_content,
            "source": metadata.get("source"),
            "modality": modality,
            "similarity_score": score,
            "metadata": metadata.copy()
        }

        textual_context.append(text_entry)

        # Load original images and video frames
        if modality in ("image", "video"):
            visual_path = (
                metadata.get("image_path")
                if modality == "image"
                else metadata.get("frame_path")
            )

            if not visual_path:
                raise ValueError(
                    f"Missing visual file path for {modality} document."
                )

            visual_file = Path(visual_path)

            if not visual_file.is_file():
                raise FileNotFoundError(
                    f"Visual file not found: {visual_file}"
                )

            # Read the original visual file
            with open(visual_file, "rb") as file:
                image_bytes = file.read()

            if not image_bytes:
                raise ValueError(f"Visual file is empty: {visual_file}")

            mime_type, _ = mimetypes.guess_type(str(visual_file))

            if not mime_type or not mime_type.startswith("image/"):
                raise ValueError(
                    f"Unsupported image format: {visual_file}"
                )

            visual_entry = {
                "rank": rank,
                "source": metadata.get("source"),
                "modality": modality,
                "timestamp_seconds": metadata.get("timestamp_seconds"),
                "image_path": str(visual_file),
                "mime_type": mime_type,
                "image_bytes": image_bytes
            }

            visual_context.append(visual_entry)

    return textual_context, visual_context


def generate_multimodal_answer(query, textual_context, visual_context):
    """Generate a grounded answer using retrieved text and original visuals."""

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if not textual_context:
        raise ValueError("Textual context cannot be empty.")

    # Prepare retrieved textual context
    context_parts = []

    for entry in textual_context:
        metadata = entry.get("metadata", {})

        context_parts.append(
            f"""
Rank: {entry['rank']}

Source: {entry['source']}

Modality: {entry['modality']}

Similarity Score: {entry['similarity_score']:.4f}

Metadata: {metadata}

Content:

{entry['content']}
"""
        )

    retrieved_text = "\n".join(context_parts)

    # Prepare the prompt for Gemini Vision
    prompt = f"""
You are a helpful assistant answering questions using retrieved
multimodal information.

USER QUESTION:

{query}

RETRIEVED TEXTUAL CONTEXT:

{retrieved_text}

INSTRUCTIONS:

1. Answer the user's question using the provided retrieved context
   and original visual content.

2. Inspect the supplied images and video frames when relevant.

3. Combine information from text and visuals when both are useful.

4. Do not invent information or make unsupported assumptions.

5. If the context does not contain enough information, clearly
   mention that.

6. Keep the answer clear, accurate, and relevant.

7. Include source references using the available filenames,
   PDF page numbers, or video timestamps whenever applicable.
"""

    # Prepare multimodal content for Gemini
    message_content = [
        {
            "type": "text",
            "text": prompt
        }
    ]

    # Attach the original retrieved images and video frames
    for entry in visual_context:
        image_base64 = base64.b64encode(
            entry["image_bytes"]
        ).decode("utf-8")

        visual_label = (
            f"Original visual from source: {entry['source']}, "
            f"modality: {entry['modality']}"
        )

        if entry["timestamp_seconds"] is not None:
            visual_label += (
                f", timestamp: {entry['timestamp_seconds']} seconds"
            )

        message_content.append(
            {
                "type": "text",
                "text": visual_label
            }
        )

        message_content.append(
            {
                "type": "image",
                "base64": image_base64,
                "mime_type": entry["mime_type"]
            }
        )

    # Generate the final answer using Gemini Vision
    response = llm.invoke(
        [
            {
                "role": "user",
                "content": message_content
            }
        ]
    )

    answer = response.text

    if answer and answer.strip():
        return answer.strip()

    # Fallback for responses returned as structured content blocks
    if isinstance(response.content, list):
        text_blocks = [
            block.get("text", "")
            for block in response.content
            if isinstance(block, dict) and block.get("type") == "text"
        ]

        answer = "\n".join(text_blocks).strip()

    if not answer:
        raise RuntimeError("Gemini returned an empty answer.")

    return answer


if __name__ == "__main__":
    # 1. Process each modality
    pdf_chunks = load_and_split_pdf()
    image_document = process_image()

    video_frames = extract_video_frames()
    video_frame_descriptions = describe_video_frames(video_frames)

    extract_video_audio()
    transcript = transcribe_video_audio()
    transcript_segments = parse_transcript_segments(transcript)

    # 2. Combine all textual representations into LangChain Documents
    multimodal_documents = prepare_multimodal_documents(
        pdf_chunks,
        image_document,
        video_frame_descriptions,
        transcript_segments
    )

    modality_counts = {}

    for document in multimodal_documents:
        modality = document.metadata["modality"]
        modality_counts[modality] = modality_counts.get(modality, 0) + 1

    print("\n========== MULTIMODAL DOCUMENTS ==========")
    print(f"Total documents: {len(multimodal_documents)}")

    for modality, count in modality_counts.items():
        print(f"{modality}: {count}")

    # 3. Generate embeddings and store them in FAISS
    embeddings_model, vectors = generate_embeddings(multimodal_documents)

    faiss_index, document_mapping = store_embeddings_in_faiss(
        vectors,
        multimodal_documents
    )

    print("\n========== FAISS VECTOR STORE ==========")
    print(f"Vectors stored: {faiss_index.ntotal}")
    print(f"Embedding dimensions: {faiss_index.d}")

    # 4. Retrieve relevant documents for the user query
    query = "What are the different chunking strategies in RAG?"

    retrieved_documents = retrieve_documents(
        query=query,
        embeddings_model=embeddings_model,
        faiss_index=faiss_index,
        document_mapping=document_mapping,
        top_k=5
    )

    print("\n========== RETRIEVAL RESULTS ==========")
    print(f"Query: {query}")

    for rank, result in enumerate(retrieved_documents, start=1):
        document = result["document"]

        print(
            f"\nRank: {rank} | Score: {result['score']:.4f} | "
            f"Modality: {document.metadata.get('modality')}"
        )

        print(f"Source: {document.metadata.get('source')}")
        print(f"Content: {document.page_content[:250]}")

    # 5. Prepare retrieved text and original visual content
    textual_context, visual_context = prepare_multimodal_context(
        retrieved_documents
    )

    print("\n========== MULTIMODAL CONTEXT ==========")
    print(f"Text entries: {len(textual_context)}")
    print(f"Original visual files: {len(visual_context)}")

    # 6. Generate an answer using retrieved text and original visuals
    answer = generate_multimodal_answer(
        query,
        textual_context,
        visual_context
    )

    print("\n========== GEMINI VISION ANSWER ==========")
    print(answer)