import sys
import os
import unittest

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from c05_vector_databases.chunking.custom_chunking_utils import custom_spliter, catch_title

class TestCustomSpliter(unittest.TestCase):
    def test_basic_chapter_splitting(self):
        """Test splitting text with Roman numeral chapters"""
        text = "Some introduction\n\nI. The Beginning\nThis is chapter one.\n\nII. The Middle\nThis is chapter two."
        result = custom_spliter(text)
        self.assertEqual(len(result), 3)
        self.assertIn("Some introduction", result[0])
        self.assertIn("I. The Beginning", result[1])
        self.assertIn("II. The Middle", result[2])
        
    def test_empty_text(self):
        """Test handling of empty text"""
        text = ""
        result = custom_spliter(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "")
        
    def test_no_matches(self):
        """Test text without Roman numeral chapter headers"""
        text = "This is just regular text\nWith multiple lines\nBut no chapter headers"
        result = custom_spliter(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], text)

    def test_single_chapter(self):
        """Test text with a single Roman numeral chapter"""
        text = "Intro text\n\nI. First Chapter\nChapter content here"
        result = custom_spliter(text)
        self.assertEqual(len(result), 2)
        self.assertIn("Intro text", result[0])
        self.assertIn("I. First Chapter", result[1])

    def test_multiple_chapters(self):
        """Test text with multiple Roman numeral chapters"""
        text = "Intro\n\nI. First\nContent 1\n\nII. Second\nContent 2\n\nIII. Third\nContent 3\n\nIV. Fourth\nContent 4"
        result = custom_spliter(text)
        # Should split into 5 parts (1 intro + 4 chapters)
        self.assertEqual(len(result), 5)
        self.assertIn("Intro", result[0])
        self.assertIn("I. First", result[1])
        self.assertIn("II. Second", result[2])
        self.assertIn("III. Third", result[3])
        self.assertIn("IV. Fourth", result[4])

    def test_different_roman_numerals(self):
        """Test various Roman numeral formats"""
        text = "Intro\n\nI. First\nContent\n\nV. Fifth\nMore content\n\nX. Tenth\nEven more"
        result = custom_spliter(text)
        self.assertEqual(len(result), 4)
        self.assertIn("Intro", result[0])
        self.assertIn("I. First", result[1])
        self.assertIn("V. Fifth", result[2])
        self.assertIn("X. Tenth", result[3])
        
    def test_edge_case_with_period(self):
        """Test edge case where period exists but not chapter format"""
        text = "This is a sentence.\nThis is another sentence."
        result = custom_spliter(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], text)
        
    def test_complex_roman_numerals(self):
        """Test complex Roman numerals including multi-character ones"""
        text = "Preface\n\nI. First\nContent\n\nIV. Fourth\nContent\n\nIX. Ninth\nContent\n\nXX. Twentieth\nContent"
        result = custom_spliter(text)
        self.assertEqual(len(result), 5)
        self.assertIn("Preface", result[0])
        self.assertIn("I. First", result[1])
        self.assertIn("IV. Fourth", result[2])
        self.assertIn("IX. Ninth", result[3])
        self.assertIn("XX. Twentieth", result[4])
        
    def test_mixed_content(self):
        """Test with mixed content that includes potential false positives"""
        text = "This is a regular sentence.\nIt talks about Mr. Smith.\n\nI. The Real Chapter\nThis is the actual chapter."
        result = custom_spliter(text)
        self.assertEqual(len(result), 2)
        self.assertIn("This is a regular sentence.", result[0])
        self.assertIn("I. The Real Chapter", result[1])
        
    def test_return_type(self):
        """Test that the function returns a list"""
        text = "Some text\n\nI. Chapter\nChapter content"
        result = custom_spliter(text)
        self.assertIsInstance(result, list)
        
    def test_large_text_performance(self):
        """Test performance with larger text"""
        # Create a large text with many chapters
        chapters = []
        for i in range(1, 51):  # 50 chapters
            roman = self.int_to_roman(i)
            chapters.append(f"\n{roman}. Chapter {i}\nThis is the content of chapter {i}." * 10)
        
        large_text = "Introduction. This is the introduction." + "".join(chapters)
        result = custom_spliter(large_text)
        self.assertGreaterEqual(len(result), 50)  # At least 50 chunks
        
    def int_to_roman(self, num):
        """Helper method to convert integer to Roman numeral"""
        val = [
            1000, 900, 500, 400,
            100, 90, 50, 40,
            10, 9, 5, 4,
            1
        ]
        syms = [
            "M", "CM", "D", "CD",
            "C", "XC", "L", "XL",
            "X", "IX", "V", "IV",
            "I"
        ]
        roman_num = ''
        i = 0
        while num > 0:
            for _ in range(num // val[i]):
                roman_num += syms[i]
                num -= val[i]
            i += 1
        return roman_num

class TestCatchTitle(unittest.TestCase):
    def test_basic_title_capture(self):
        """Test capturing a title from text with Roman numeral header"""
        text = "I. THE BEGINNING\r\nThis is the content of the first chapter."
        result = catch_title(text)
        self.assertEqual(result, "THE BEGINNING")
        
    def test_lowercase_roman_numerals(self):
        """Test that lowercase Roman numerals are not captured"""
        text = "i. the beginning\r\nContent here"
        result = catch_title(text)
        self.assertIsNone(result)
        
    def test_mixed_case_title(self):
        """Test that titles with mixed case are not captured"""
        text = "II. The Middle\r\nContent here"
        result = catch_title(text)
        self.assertIsNone(result)
        
    def test_no_match(self):
        """Test text that doesn't match the pattern"""
        text = "This is just regular text\r\nWith no Roman numeral headers"
        result = catch_title(text)
        self.assertIsNone(result)
        
    def test_empty_text(self):
        """Test handling of empty text"""
        text = ""
        result = catch_title(text)
        self.assertIsNone(result)
        
    def test_complex_title(self):
        """Test capturing a complex title with spaces and dashes"""
        text = "V. THE END OF THE STORY - CONCLUSION\r\nFinal chapter content."
        result = catch_title(text)
        self.assertEqual(result, "THE END OF THE STORY - CONCLUSION")
        
    def test_different_roman_numerals(self):
        """Test various Roman numeral patterns"""
        test_cases = [
            ("I. SINGLE TITLE\r\n", "SINGLE TITLE"),
            ("V. FIFTH CHAPTER\r\n", "FIFTH CHAPTER"),
            ("X. TENTH PART\r\n", "TENTH PART"),
            ("L. FIFTIETH SECTION\r\n", "FIFTIETH SECTION"),
            ("C. HUNDREDTH CHAPTER\r\n", "HUNDREDTH CHAPTER"),
            ("D. FIVE HUNDRED\r\n", "FIVE HUNDRED"),
            ("M. THOUSAND SECTION\r\n", "THOUSAND SECTION"),
            ("IV. FOURTH TITLE\r\n", "FOURTH TITLE"),
            ("IX. NINTH CHAPTER\r\n", "NINTH CHAPTER"),
            ("XL. FORTIETH PART\r\n", "FORTIETH PART"),
            ("XC. NINETIETH SECTION\r\n", "NINETIETH SECTION"),
            ("CD. FOUR HUNDRED\r\n", "FOUR HUNDRED"),
            ("CM. NINE HUNDRED\r\n", "NINE HUNDRED"),
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = catch_title(text)
                self.assertEqual(result, expected)
                
    def test_partial_match(self):
        """Test that partial matches don't interfere"""
        text = "This is some text with I. in the middle but not at the beginning\r\n"
        result = catch_title(text)
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()