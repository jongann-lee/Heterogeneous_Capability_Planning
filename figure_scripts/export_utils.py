"""Shared PDF and proportionally sized SVG export for publication figures."""

from io import BytesIO
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib as mpl


SVG_WIDTH = 600


def save_pdf_and_svg(figure, pdf_path, *, metadata=None, dpi=300, svg_width=SVG_WIDTH):
    """Save a PDF and a 600 px-wide SVG with the same basename.

    PDF dimensions remain unchanged. SVG text stays text, rasterized plotting
    artists become vectors, and genuine image backgrounds stay embedded images.
    The original SVG viewBox and an explicit preserveAspectRatio keep the full
    plot proportional. Both files are replaced when the scripts are rerun.
    """
    if not math.isfinite(svg_width) or svg_width <= 0:
        raise ValueError("SVG width must be finite and positive")
    pdf_path = Path(pdf_path)
    svg_path = pdf_path.with_suffix(".svg")
    metadata = dict(metadata or {})
    pdf_buffer = BytesIO()
    figure.savefig(pdf_buffer, format="pdf", dpi=dpi, bbox_inches=None, metadata=metadata)

    # Keep the compact rasterized terrain in PDFs, but expose its individual
    # cells as editable vectors in SVG. Restore the figure even if export fails.
    rasterized_artists = [artist for artist in figure.findobj() if artist.get_rasterized()]
    svg_buffer = BytesIO()
    try:
        for artist in rasterized_artists:
            artist.set_rasterized(False)
        svg_metadata = {"Title": metadata.get("Title"),
                        "Description": metadata.get("Subject"),
                        "Creator": metadata.get("Creator")}
        with mpl.rc_context({"svg.fonttype": "none", "svg.image_inline": True}):
            figure.savefig(svg_buffer, format="svg", dpi=dpi, bbox_inches=None,
                           metadata=svg_metadata)
    finally:
        for artist in rasterized_artists:
            artist.set_rasterized(True)

    root = ET.fromstring(svg_buffer.getvalue())
    _, _, view_width, view_height = map(float, root.attrib["viewBox"].split())
    root.set("width", f"{svg_width:g}px")
    root.set("height", f"{svg_width * view_height / view_width:.10g}px")
    root.set("preserveAspectRatio", "xMidYMid meet")
    for prefix, uri in (
        ("", "http://www.w3.org/2000/svg"),
        ("xlink", "http://www.w3.org/1999/xlink"),
        ("dc", "http://purl.org/dc/elements/1.1/"),
        ("cc", "http://creativecommons.org/ns#"),
        ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
    ):
        ET.register_namespace(prefix, uri)
    svg_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    with pdf_path.open("wb") as stream:
        stream.write(pdf_buffer.getvalue())
    with svg_path.open("wb") as stream:
        stream.write(svg_bytes)
    return svg_path
