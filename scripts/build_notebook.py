from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "Fingerprint_Forensic_Preprocessing_Reconstruction.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(
        """
        # Fingerprint Image Preprocessing and Reconstruction Pipeline

        This notebook is built for forensic fingerprint preprocessing. It preserves the original evidence image, shows every intermediate stage, and keeps artifact masks/provenance visible whenever any pixel is suppressed or reconstructed.

        Forensic guardrails:
        - The original image is never overwritten.
        - Automatic artifact removal is conservative.
        - Yellow overlay means review-only candidate, not removed.
        - Blue overlay means a small supported gap was reconstructed.
        - Red overlay means an artifact was detected but left unchanged because reconstruction would be unsafe.
        - The enhanced and binary views are analysis aids, not replacement evidence.
        """
    ),
    code(
        """
        from pathlib import Path
        import ast
        import warnings

        import matplotlib.pyplot as plt
        import pandas as pd
        from IPython.display import Markdown, display

        from fingerprint_pipeline import (
            FingerprintConfig,
            contact_sheet,
            discover_images,
            display_comparison,
            display_stage_grid,
            mask_overlay,
            metrics_frame,
            process_batch,
            provenance_overlay,
            run_pipeline,
            save_pipeline_outputs,
        )

        warnings.filterwarnings("ignore", category=FutureWarning)
        plt.rcParams["figure.dpi"] = 120

        PROJECT_ROOT = Path.cwd()
        cfg = FingerprintConfig(output_dir="outputs")
        image_paths = discover_images(cfg.input_dirs)
        print(f"Project: {PROJECT_ROOT}")
        print(f"Images found: {len(image_paths)}")
        for path in image_paths:
            print(" -", path)
        """
    ),
    md(
        """
        ## Input Review

        Confirm that the notebook sees the fingerprint samples. The pipeline accepts both `utils/image` and the current `utils` layout.
        """
    ),
    code(
        """
        if not image_paths:
            raise FileNotFoundError("No images found under utils/image or utils.")

        sheet = contact_sheet(image_paths, thumb_size=(220, 180), cols=4)
        plt.figure(figsize=(12, 8))
        plt.imshow(sheet)
        plt.axis("off")
        plt.title("Available fingerprint samples")
        plt.show()
        """
    ),
    md(
        """
        ## Single-Image Forensic Pipeline

        Edit the configuration below for repeatable experiments. Manual masks are intentionally explicit:
        - `manual_rectangles`: `(x, y, width, height)`
        - `manual_lines`: `(x1, y1, x2, y2, width)`
        - `manual_polygons`: `[(x1, y1), (x2, y2), ...]`
        """
    ),
    code(
        """
        selected_image = image_paths[0]

        cfg = FingerprintConfig(
            output_dir="outputs",
            illumination_sigma=26.0,
            clahe_clip_limit=2.0,
            segmentation_block_size=23,
            denoise_median_size=3,
            denoise_bilateral_diameter=5,
            remove_ambiguous_dark_inside_roi=False,
            max_inpaint_component_area=1600,
            max_inpaint_component_width=48,
            max_inpaint_component_height=48,
            ridge_period_px=9.0,
            gabor_blend=0.32,
            suppress_background=True,
            manual_rectangles=[],
            manual_lines=[],
            manual_polygons=[],
        )

        result = run_pipeline(selected_image, cfg)
        display(Markdown(f"### Case: `{selected_image}`"))
        display_comparison(result)
        display_stage_grid(result, max_cols=3, figsize=(15, 12))
        display(metrics_frame(result))
        if result.warnings:
            display(Markdown("### Review Notes"))
            for warning in result.warnings:
                display(Markdown(f"- {warning}"))
        """
    ),
    md(
        """
        ## Live Parameter Tuning

        Use this panel for live experiments. Sliders update the full stage grid and side-by-side comparison. Keep `Remove ambiguous dark ROI marks` off unless you have reviewed the yellow mask and accepted the risk.
        """
    ),
    code(
        """
        import ipywidgets as widgets

        image_dropdown = widgets.Dropdown(
            options=[(p.name, str(p)) for p in image_paths],
            value=str(selected_image),
            description="Image",
            layout=widgets.Layout(width="420px"),
        )
        illumination = widgets.FloatSlider(value=26.0, min=5.0, max=65.0, step=1.0, description="Illumination", continuous_update=False)
        clahe_clip = widgets.FloatSlider(value=2.0, min=0.5, max=5.0, step=0.1, description="CLAHE", continuous_update=False)
        seg_block = widgets.IntSlider(value=23, min=9, max=61, step=2, description="ROI block", continuous_update=False)
        median_size = widgets.SelectionSlider(options=[1, 3, 5], value=3, description="Median", continuous_update=False)
        bilateral_d = widgets.SelectionSlider(options=[1, 3, 5, 7, 9], value=5, description="Bilateral", continuous_update=False)
        remove_dark = widgets.Checkbox(value=False, description="Remove ambiguous dark ROI marks")
        max_area = widgets.IntSlider(value=1600, min=50, max=8000, step=50, description="Max repair area", continuous_update=False)
        max_width = widgets.IntSlider(value=48, min=8, max=160, step=4, description="Max repair width", continuous_update=False)
        ridge_period = widgets.FloatSlider(value=9.0, min=4.0, max=18.0, step=0.5, description="Ridge period", continuous_update=False)
        gabor_blend = widgets.FloatSlider(value=0.32, min=0.0, max=0.65, step=0.02, description="Gabor blend", continuous_update=False)
        suppress_bg = widgets.Checkbox(value=True, description="Suppress background in final view")
        manual_rectangles = widgets.Textarea(value="[]", description="Rectangles", layout=widgets.Layout(width="720px", height="60px"))
        manual_lines = widgets.Textarea(value="[]", description="Lines", layout=widgets.Layout(width="720px", height="60px"))
        manual_polygons = widgets.Textarea(value="[]", description="Polygons", layout=widgets.Layout(width="720px", height="60px"))

        def parse_literal(text, fallback):
            try:
                value = ast.literal_eval(text)
                return value if isinstance(value, list) else fallback
            except Exception:
                return fallback

        def live_run(
            image,
            illumination_sigma,
            clahe_clip_limit,
            segmentation_block_size,
            denoise_median_size,
            denoise_bilateral_diameter,
            remove_ambiguous_dark_inside_roi,
            max_inpaint_component_area,
            max_inpaint_component_width,
            ridge_period_px,
            gabor_blend,
            suppress_background,
            rectangles,
            lines,
            polygons,
        ):
            live_cfg = FingerprintConfig(
                output_dir="outputs",
                illumination_sigma=illumination_sigma,
                clahe_clip_limit=clahe_clip_limit,
                segmentation_block_size=segmentation_block_size,
                denoise_median_size=denoise_median_size,
                denoise_bilateral_diameter=denoise_bilateral_diameter,
                remove_ambiguous_dark_inside_roi=remove_ambiguous_dark_inside_roi,
                max_inpaint_component_area=max_inpaint_component_area,
                max_inpaint_component_width=max_inpaint_component_width,
                max_inpaint_component_height=max_inpaint_component_width,
                ridge_period_px=ridge_period_px,
                gabor_blend=gabor_blend,
                suppress_background=suppress_background,
                manual_rectangles=parse_literal(rectangles, []),
                manual_lines=parse_literal(lines, []),
                manual_polygons=parse_literal(polygons, []),
            )
            live_result = run_pipeline(image, live_cfg)
            display(Markdown(f"### Live Case: `{image}`"))
            display_comparison(live_result)
            display_stage_grid(live_result, max_cols=3, figsize=(15, 12))
            display(metrics_frame(live_result))
            if live_result.warnings:
                display(Markdown("### Review Notes"))
                for warning in live_result.warnings:
                    display(Markdown(f"- {warning}"))

        controls = widgets.VBox([
            image_dropdown,
            widgets.HBox([illumination, clahe_clip, seg_block]),
            widgets.HBox([median_size, bilateral_d, ridge_period, gabor_blend]),
            widgets.HBox([max_area, max_width]),
            widgets.HBox([suppress_bg, remove_dark]),
            manual_rectangles,
            manual_lines,
            manual_polygons,
        ])

        output = widgets.interactive_output(
            live_run,
            {
                "image": image_dropdown,
                "illumination_sigma": illumination,
                "clahe_clip_limit": clahe_clip,
                "segmentation_block_size": seg_block,
                "denoise_median_size": median_size,
                "denoise_bilateral_diameter": bilateral_d,
                "remove_ambiguous_dark_inside_roi": remove_dark,
                "max_inpaint_component_area": max_area,
                "max_inpaint_component_width": max_width,
                "ridge_period_px": ridge_period,
                "gabor_blend": gabor_blend,
                "suppress_background": suppress_bg,
                "rectangles": manual_rectangles,
                "lines": manual_lines,
                "polygons": manual_polygons,
            },
        )
        display(controls, output)
        """
    ),
    md(
        """
        ## Mask Inspection

        Run this cell after creating `result` or a `live_result` to inspect individual mask layers. The provenance overlay is the fastest forensic review view.
        """
    ),
    code(
        """
        mask_names = [
            "roi",
            "artifact_remove",
            "artifact_review",
            "artifact_color",
            "artifact_lines",
            "artifact_dark_auto",
            "reconstructed_pixels",
            "blocked_artifact_pixels",
        ]

        fig, axes = plt.subplots(2, 4, figsize=(15, 7))
        base = result.stages["05_ridge_preserving_denoise"]
        for ax, name in zip(axes.ravel(), mask_names):
            ax.imshow(mask_overlay(base, result.masks[name]))
            ax.set_title(name)
            ax.axis("off")
        fig.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## Save Current Case

        Saves all stage images, key masks, provenance overlay, and quality metrics under `outputs/<image_stem>/`.
        """
    ),
    code(
        """
        saved = save_pipeline_outputs(result, cfg.output_dir)
        print(f"Saved {len(saved)} artifacts under {Path(cfg.output_dir) / selected_image.stem}")
        """
    ),
    md(
        """
        ## Batch Processing

        Batch mode uses the same conservative defaults and writes per-image stage outputs for audit.
        """
    ),
    code(
        """
        batch_cfg = FingerprintConfig(output_dir="outputs")
        batch_summary = process_batch(image_paths, batch_cfg, save_outputs=True)
        display(batch_summary)
        batch_summary.to_csv(Path(batch_cfg.output_dir) / "batch_quality_summary.csv", index=False)
        print("Batch summary:", Path(batch_cfg.output_dir) / "batch_quality_summary.csv")
        """
    ),
    md(
        """
        ## Parameter Tuning Guidelines

        - Increase `Illumination` for uneven scans, wet fingerprints, or shadowed document photos.
        - Reduce `CLAHE` if ridges become harsh or noise starts looking like ridge detail.
        - Increase `ROI block` for large inked/scanned impressions; reduce it for small partial prints.
        - Keep median filtering at `3` for salt-and-pepper noise; use `1` when minutiae appear softened.
        - Keep `Remove ambiguous dark ROI marks` disabled unless the yellow mask has been reviewed.
        - Raise repair limits only for narrow, verified artifacts. Large red regions should remain blocked rather than hallucinated.
        - Use manual rectangles/lines/polygons for black labels, stamps, or handwriting that automation cannot separate from ridge flow.
        - Use the original image, masks, and provenance overlay together for forensic interpretation.
        """
    ),
]


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python (.venv fingerprint)",
            "language": "python",
            "name": "fingerprint-forensic",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK)
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
