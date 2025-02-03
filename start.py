import subprocess
import sys
import os
import requests
import time 
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add at the start of the file, before other code
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_DIR)  # Change working directory to script location

def check_requirements():
    """Check if all required components are available"""
    try:
        # Use absolute path for Data directory
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(ROOT_DIR, 'Data')  # Note the capital 'D' in Data
        
        # Check if Data directory exists
        if not os.path.exists(DATA_DIR):
            logger.error(f"Data directory not found at: {DATA_DIR}")
            logger.info("Creating Data directory...")
            os.makedirs(DATA_DIR)
            logger.info("Please add your medical PDFs to the Data directory")
            return False
        
        # Check if data directory has PDFs
        pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
        if not pdf_files:
            logger.error(f"No PDF files found in {DATA_DIR}. Please add some medical PDFs.")
            return False

        # Check if models directory exists
        models_dir = os.path.join(ROOT_DIR, 'models')
        if not os.path.exists(models_dir):
            logger.info("Creating 'models' directory...")
            os.makedirs(models_dir)

        # Check if ingest.py exists
        if not os.path.exists(os.path.join(ROOT_DIR, 'ingest.py')):
            logger.error("ingest.py not found in the current directory")
            return False

        # Check if rag.py exists
        if not os.path.exists(os.path.join(ROOT_DIR, 'rag.py')):
            logger.error("rag.py not found in the current directory")
            return False

        # Check if Qdrant is running
        try:
            response = requests.get("http://localhost:6333/collections")
            if response.status_code != 200:
                raise Exception("Qdrant not responding properly")
        except Exception as e:
            logger.error("Error: Qdrant is not running!")
            logger.info("Please start Qdrant with: docker run -p 6333:6333 qdrant/qdrant")
            return False

        return True
        
    except Exception as e:
        logger.error(f"Error during requirements check: {str(e)}")
        return False

def main():
    try:
        # Check all requirements
        if not check_requirements():
            logger.error("Requirements check failed. Please fix the issues above and try again.")
            return

        # Import ingest only after requirements are checked
        try:
            from ingest import ingest_docs
        except ImportError as e:
            logger.error(f"Failed to import ingest_docs: {str(e)}")
            return

        # Ingest documents
        logger.info("Ingesting documents...")
        if not ingest_docs():
            logger.error("Failed to ingest documents. Please check the data directory.")
            return

        # Start the FastAPI server
        logger.info("Starting the application...")
        subprocess.run(["uvicorn", "rag:app", "--reload"], check=True)
        
    except KeyboardInterrupt:
        logger.info("\nShutting down gracefully...")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error starting the server: {str(e)}")
        logger.info("Make sure all requirements are installed.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()