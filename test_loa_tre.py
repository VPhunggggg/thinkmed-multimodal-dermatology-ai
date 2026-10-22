# Tạo file test, phát thẳng, nghe chữ đầu

import edge_tts, asyncio
asyncio.run(edge_tts.Communicate('Một hai ba bốn năm sáu bảy tám', 'vi-VN-HoaiMyNeural').save('/tmp/test.mp3'))

ffplay -nodisp -autoexit -loglevel quiet /tmp/test.mp3
