import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from c05_vector_databases.capstone_project.data_prep import create_database
from langchain.schema import Document


class TestCreateDatabase(unittest.TestCase):
    """Test suite for create_database function"""
    
    def test_create_database_basic(self):
        """Test create_database with mocked dependencies"""
        # Setup mocks
        mock_vector_store = MagicMock()
        
        # Mock dataset
        mock_dataset = [
            {
                "fullplot": "A test movie plot about a hero saving the world",
                "title": "Test Movie",
                "poster": "http://test.com/poster.jpg",
                "genres": ["Action", "Adventure"],
                "imdb": {"rating": 8.5}
            },
            {
                "fullplot": None,  # Should be skipped
                "title": "Movie without plot",
                "poster": None,
                "genres": None,
                "imdb": {"rating": None}
            },
            {
                "fullplot": "Another test movie plot",
                "title": None,
                "poster": None,
                "genres": ["Drama"],
                "imdb": {"rating": 7.0}
            }
        ]
        
        with patch('c05_vector_databases.capstone_project.data_prep.load_dataset') as mock_load, \
             patch('c05_vector_databases.capstone_project.data_prep.RecursiveCharacterTextSplitter') as mock_splitter_class:
            
            # Configure mocks
            mock_load.return_value = mock_dataset
            mock_splitter = MagicMock()
            mock_splitter.split_documents.return_value = [Document(page_content="Chunked content")]
            mock_splitter_class.return_value = mock_splitter
            
            # Execute
            create_database(mock_vector_store)
            
            # Verify dataset was loaded
            mock_load.assert_called_once_with("MongoDB/embedded_movies", split="train")
            
            # Verify text splitter was created with correct parameters
            mock_splitter_class.assert_called_once_with(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", ",", " ", ""]
            )
            
            # Verify add_documents was called
            mock_splitter.split_documents.assert_called_once()
            mock_vector_store.add_documents.assert_called_once_with(documents=[Document(page_content="Chunked content")])
    
    def test_create_database_filters_none_fullplot(self):
        """Test that items with None fullplot are filtered out"""
        # Setup
        mock_vector_store = MagicMock()
        
        mock_dataset = [
            {
                "fullplot": "Valid plot",
                "title": "Movie 1",
                "poster": "poster1.jpg",
                "genres": ["Action"],
                "imdb": {"rating": 5.0}
            },
            {
                "fullplot": None,  # Should be skipped
                "title": "Movie 2",
                "poster": "poster2.jpg",
                "genres": ["Drama"],
                "imdb": {"rating": 6.0}
            }
        ]
        
        with patch('c05_vector_databases.capstone_project.data_prep.load_dataset') as mock_load, \
             patch('c05_vector_databases.capstone_project.data_prep.RecursiveCharacterTextSplitter') as mock_splitter_class:
            
            mock_load.return_value = mock_dataset
            mock_splitter = MagicMock()
            mock_splitter_class.return_value = mock_splitter
            
            # Execute
            create_database(mock_vector_store)
            
            # Verify split_documents was called with only one document (the one with fullplot)
            call_args = mock_splitter.split_documents.call_args
            documents = call_args[0][0]  # First positional argument
            
            # Should only have 1 document (the one with fullplot)
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].page_content, "Valid plot")
    
    def test_create_database_handles_missing_metadata(self):
        """Test that missing metadata fields are handled correctly"""
        # Setup
        mock_vector_store = MagicMock()
        
        mock_dataset = [
            {
                "fullplot": "A plot",
                "title": None,
                "poster": None,
                "genres": None,
                "imdb": {"rating": None}
            }
        ]
        
        with patch('c05_vector_databases.capstone_project.data_prep.load_dataset') as mock_load, \
             patch('c05_vector_databases.capstone_project.data_prep.RecursiveCharacterTextSplitter') as mock_splitter_class:
            
            mock_load.return_value = mock_dataset
            mock_splitter = MagicMock()
            mock_splitter_class.return_value = mock_splitter
            
            # Execute
            create_database(mock_vector_store)
            
            # Verify the document was created with empty strings for missing values
            call_args = mock_splitter.split_documents.call_args
            documents = call_args[0][0]
            
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].page_content, "A plot")
            self.assertEqual(documents[0].metadata["title"], "")
            self.assertEqual(documents[0].metadata["poster"], "")
            self.assertEqual(documents[0].metadata["genre"], "")
            self.assertEqual(documents[0].metadata["imdb_rating"], "")
    
    def test_create_database_genres_joined_with_semicolon(self):
        """Test that genres are joined with semicolon"""
        # Setup
        mock_vector_store = MagicMock()
        
        mock_dataset = [
            {
                "fullplot": "A plot",
                "title": "Movie",
                "poster": "poster.jpg",
                "genres": ["Action", "Adventure", "Thriller"],
                "imdb": {"rating": 7.5}
            }
        ]
        
        with patch('c05_vector_databases.capstone_project.data_prep.load_dataset') as mock_load, \
             patch('c05_vector_databases.capstone_project.data_prep.RecursiveCharacterTextSplitter') as mock_splitter_class:
            
            mock_load.return_value = mock_dataset
            mock_splitter = MagicMock()
            mock_splitter_class.return_value = mock_splitter
            
            # Execute
            create_database(mock_vector_store)
            
            # Verify genres are joined with semicolon
            call_args = mock_splitter.split_documents.call_args
            documents = call_args[0][0]
            
            self.assertEqual(documents[0].metadata["genre"], "Action; Adventure; Thriller")
    
    def test_create_database_empty_dataset(self):
        """Test create_database with empty dataset"""
        # Setup
        mock_vector_store = MagicMock()
        
        mock_dataset = []
        
        with patch('c05_vector_databases.capstone_project.data_prep.load_dataset') as mock_load, \
             patch('c05_vector_databases.capstone_project.data_prep.RecursiveCharacterTextSplitter') as mock_splitter_class:
            
            mock_load.return_value = mock_dataset
            mock_splitter = MagicMock()
            mock_splitter_class.return_value = mock_splitter
            
            # Execute
            create_database(mock_vector_store)
            
            # Verify split_documents was called with empty list
            mock_splitter.split_documents.assert_called_once_with([])
    
    def test_create_database_all_items_none_fullplot(self):
        """Test when all items have None fullplot"""
        # Setup
        mock_vector_store = MagicMock()
        
        mock_dataset = [
            {"fullplot": None, "title": "Movie 1"},
            {"fullplot": None, "title": "Movie 2"}
        ]
        
        with patch('c05_vector_databases.capstone_project.data_prep.load_dataset') as mock_load, \
             patch('c05_vector_databases.capstone_project.data_prep.RecursiveCharacterTextSplitter') as mock_splitter_class:
            
            mock_load.return_value = mock_dataset
            mock_splitter = MagicMock()
            mock_splitter_class.return_value = mock_splitter
            
            # Execute
            create_database(mock_vector_store)
            
            # Verify split_documents was called with empty list
            mock_splitter.split_documents.assert_called_once_with([])
    
    def test_create_database_chunks_added_to_vector_store(self):
        """Test that chunked documents are added to vector store"""
        # Setup
        mock_vector_store = MagicMock()
        
        mock_dataset = [
            {
                "fullplot": "First movie plot",
                "title": "First Movie",
                "poster": "first.jpg",
                "genres": ["Action"],
                "imdb": {"rating": 8.0}
            }
        ]
        
        # Create expected chunks
        chunk1 = Document(page_content="Chunk 1 from first movie", metadata={"title": "First Movie"})
        chunk2 = Document(page_content="Chunk 2 from first movie", metadata={"title": "First Movie"})
        
        with patch('c05_vector_databases.capstone_project.data_prep.load_dataset') as mock_load, \
             patch('c05_vector_databases.capstone_project.data_prep.RecursiveCharacterTextSplitter') as mock_splitter_class:
            
            mock_load.return_value = mock_dataset
            mock_splitter = MagicMock()
            mock_splitter.split_documents.return_value = [chunk1, chunk2]
            mock_splitter_class.return_value = mock_splitter
            
            # Execute
            create_database(mock_vector_store)
            
            # Verify all chunks were added to vector store
            mock_vector_store.add_documents.assert_called_once_with(documents=[chunk1, chunk2])
            
            # Verify the chunks have correct content
            called_chunks = mock_vector_store.add_documents.call_args[1]['documents']
            self.assertEqual(len(called_chunks), 2)
            self.assertEqual(called_chunks[0].page_content, "Chunk 1 from first movie")
            self.assertEqual(called_chunks[1].page_content, "Chunk 2 from first movie")
    
    def test_create_database_single_movie(self):
        """Test create_database with a single movie entry"""
        # Setup
        mock_vector_store = MagicMock()
        
        mock_dataset = [
            {
                "fullplot": "A hero's journey through a mystical land",
                "title": "The Mystic Quest",
                "poster": "https://example.com/mystic.jpg",
                "genres": ["Fantasy", "Adventure"],
                "imdb": {"rating": 8.7}
            }
        ]
        
        with patch('c05_vector_databases.capstone_project.data_prep.load_dataset') as mock_load, \
             patch('c05_vector_databases.capstone_project.data_prep.RecursiveCharacterTextSplitter') as mock_splitter_class:
            
            mock_load.return_value = mock_dataset
            mock_splitter = MagicMock()
            mock_splitter.split_documents.return_value = [Document(page_content="Single chunk")]
            mock_splitter_class.return_value = mock_splitter
            
            # Execute
            create_database(mock_vector_store)
            
            # Verify only one document was processed
            call_args = mock_splitter.split_documents.call_args
            documents = call_args[0][0]
            
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].page_content, "A hero's journey through a mystical land")
            self.assertEqual(documents[0].metadata["title"], "The Mystic Quest")
            self.assertEqual(documents[0].metadata["poster"], "https://example.com/mystic.jpg")
            self.assertEqual(documents[0].metadata["genre"], "Fantasy; Adventure")
            self.assertEqual(documents[0].metadata["imdb_rating"], 8.7)


if __name__ == '__main__':
    unittest.main()