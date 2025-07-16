from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

# Nếu collection 'drant_test' tồn tại:
if client.collection_exists(collection_name="startups"):
    client.delete_collection(collection_name="startups")
    print("Collection startups đã tồn tại và sẽ bị xoá.")
    print("Collection dranstartupst_test đã bị xoá.")
else:
    print("Collection startups không tồn tại hoặc đã bị xoá trước rồi.")