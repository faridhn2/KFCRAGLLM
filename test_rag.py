from main import KFCRAG
import unittest
class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.kfc_rag = KFCRAG()
        
    def test_query1(self):
        q = 'Hi, do you have cola?'
        result = self.kfc_rag.query(q)
        outp = 'pepsi' in result.lower()
        self.assertEqual(outp, True)
    
    def test_query2(self):
        q = 'Give me a Veggie Tender, medium, with salad'
        result = self.kfc_rag.query(q)
        outp = '4 veggie tender meal' in result.lower()
        self.assertEqual(outp, True)
    
    def test_query3(self):
        q = 'Give me an orange chocolate milkshake, medium'
        result = self.kfc_rag.query(q)
        outp = 'sorry' in result.lower()
        self.assertEqual(outp, True)

    def test_query4(self):
        q = 'Give me the gluten free burger options'
        result = self.kfc_rag.query(q)
        outp = 'no ' in result.lower()
        self.assertEqual(outp, True)
    
    def test_query5(self):
        q = 'How many calories does the Colonel have?'
        result = self.kfc_rag.query(q)
        outp = '150' in result.lower()
        self.assertEqual(outp, True)
    
    def test_query6(self):
        q = 'Can I get a Whopper?'
        result = self.kfc_rag.query(q)
        outp = 'no' in result.lower()
        self.assertEqual(outp, True)
    
if __name__ == '__main__':
    unittest.main()
