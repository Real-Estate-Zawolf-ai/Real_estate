import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.helpers.config import get_settings
from app.database.vector_db import create_client
from app.rag.retrieval.hybrid_search import embed_query, _to_qdrant_filter
from app.rag.retrieval.metadata_filter import build_metadata_filter

s = get_settings()
client = create_client()
q = "Apartment for sale in New Cairo, 3 bedrooms, budget 20 million EGP"
req = {'property_type': 'Apartment', 'district': 'New Cairo', 'bedrooms': 3, 'transaction_type': 'sale', 'max_price': 20000000.0}
filt = build_metadata_filter(req)
print("filter", filt.describe())
qf = _to_qdrant_filter(filt)
print("qdrant filter", qf)
vec = embed_query(q)
pts = client.search(collection_name=s.QDRANT_COLLECTION, query_vector=vec, query_filter=qf, limit=5)
print("count", len(pts))
client.close()
