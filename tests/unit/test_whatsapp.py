from app.channels.whatsapp import extract_incoming_messages


def test_extracts_text_and_audio_messages():
    payload = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [
                        {"id": "m1", "from": "20100", "type": "text", "text": {"body": "Find a villa"}},
                        {"id": "m2", "from": "20100", "type": "audio", "audio": {"id": "media-1"}},
                    ]
                }
            }]
        }]
    }

    assert extract_incoming_messages(payload) == [
        {"id": "m1", "from": "20100", "type": "text", "text": "Find a villa"},
        {"id": "m2", "from": "20100", "type": "audio", "media_id": "media-1"},
    ]
