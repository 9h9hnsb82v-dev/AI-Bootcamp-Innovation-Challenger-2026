"""회귀 방지: AI-Hub bbox는 EXIF 회전 적용 좌표계이므로 이미지 로딩은 항상 EXIF를 적용해야 한다."""
import io

from PIL import Image

from ml.vision.imageio import open_oriented


def _jpeg(size, orientation, color=(255, 0, 0)):
    im = Image.new("RGB", size, color)
    ex = im.getexif()
    ex[274] = orientation
    b = io.BytesIO()
    im.save(b, "JPEG", exif=ex.tobytes())
    return b.getvalue()


def test_exif_rotation_applied():
    assert open_oriented(_jpeg((40, 20), 6)).size == (20, 40)
    assert open_oriented(_jpeg((40, 20), 1)).size == (40, 20)


def test_orientation3_flips_pixels():
    im = Image.new("RGB", (4, 2), (0, 0, 0))
    im.putpixel((0, 0), (255, 255, 255))
    ex = im.getexif()
    ex[274] = 3
    b = io.BytesIO()
    im.save(b, "PNG", exif=ex.tobytes())
    out = open_oriented(b.getvalue())
    assert out.getpixel((3, 1))[0] > 200 and out.getpixel((0, 0))[0] < 50
