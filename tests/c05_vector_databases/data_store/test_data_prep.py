import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from c05_vector_databases.data_store.data_prep import create_chunks
from langchain.schema import Document


class TestCreateChunks(unittest.TestCase):
    """Test suite for create_chunks function"""
    
    def test_create_chunks_basic(self):
        """Test basic chunking with a real file"""
        # Test with one of the existing data files
        file_name = "StudyInScarlet.txt"
        
        # This will only work if the file exists
        if os.path.exists(os.path.join(
            os.path.dirname(__file__), 
            '..', '..', '..', 
            'c05_vector_databases', 'data', file_name
        )):
            chunks = create_chunks(file_name)
            
            # Verify we get a list of Document objects
            self.assertIsInstance(chunks, list)
            self.assertGreater(len(chunks), 0)
            
            # Verify each chunk is a Document
            for chunk in chunks:
                self.assertIsInstance(chunk, Document)
                
            # Verify chunks have content
            self.assertGreater(len(chunks[0].page_content), 0)
        else:
            self.skipTest(f"Test file {file_name} not found")
    
    @patch('c05_vector_databases.data_store.data_prep.TextLoader')
    @patch('c05_vector_databases.data_store.data_prep.RecursiveCharacterTextSplitter')
    def test_create_chunks_splitter_config(self, mock_splitter_class, mock_loader_class):
        """Test that the text splitter is configured correctly"""
        # Setup mocks
        mock_loader = MagicMock()
        mock_loader.load.return_value = [Document(page_content="Sample text")]
        mock_loader_class.return_value = mock_loader
        
        mock_splitter = MagicMock()
        mock_splitter.split_documents.return_value = [
            Document(page_content="Chunk 1"),
            Document(page_content="Chunk 2")
        ]
        mock_splitter_class.return_value = mock_splitter
        
        # Execute
        result = create_chunks("test.txt")
        
        # Verify TextLoader was called with correct encoding
        mock_loader_class.assert_called_once()
        call_args = mock_loader_class.call_args
        self.assertEqual(call_args.kwargs.get('encoding'), 'utf8')
        
        # Verify splitter was configured correctly
        mock_splitter_class.assert_called_once_with(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ".", "!", ",", ""],
        )
        
        # Verify split_documents was called
        mock_splitter.split_documents.assert_called_once()
        
        # Verify result
        self.assertEqual(len(result), 2)
    
    @patch('c05_vector_databases.data_store.data_prep.TextLoader')
    def test_create_chunks_file_path_construction(self, mock_loader_class):
        """Test that file path is constructed correctly"""
        mock_loader = MagicMock()
        mock_loader.load.return_value = [Document(page_content="Test")]
        mock_loader_class.return_value = mock_loader
        
        file_name = "test_file.txt"
        create_chunks(file_name)
        
        # Verify the file path construction
        call_args = mock_loader_class.call_args[0][0]
        self.assertTrue(call_args.endswith(os.path.join('data', file_name)))
        self.assertIn('c05_vector_databases', call_args)
    
    @patch('c05_vector_databases.data_store.data_prep.TextLoader')
    @patch('c05_vector_databases.data_store.data_prep.RecursiveCharacterTextSplitter')
    def test_create_chunks_returns_document_list(self, mock_splitter_class, mock_loader_class):
        """Test that the function returns a list of Document objects"""
        # Setup
        mock_loader = MagicMock()
        mock_loader.load.return_value = [Document(page_content="Test content")]
        mock_loader_class.return_value = mock_loader
        
        mock_splitter = MagicMock()
        expected_docs = [
            Document(page_content="Chunk 1", metadata={"source": "test"}),
            Document(page_content="Chunk 2", metadata={"source": "test"}),
            Document(page_content="Chunk 3", metadata={"source": "test"})
        ]
        mock_splitter.split_documents.return_value = expected_docs
        mock_splitter_class.return_value = mock_splitter
        
        # Execute
        result = create_chunks("test.txt")
        
        # Verify
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        for doc in result:
            self.assertIsInstance(doc, Document)
        self.assertEqual(result, expected_docs)


if __name__ == '__main__':
    unittest.main()
