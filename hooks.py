from mkdocs.structure.files import File

_DOC_ASSETS = (
    "assets/diagrams/architecture-overview.png",
    "assets/diagrams/automatic-index-planning.png",
    "assets/diagrams/geometry-layout.png",
    "assets/diagrams/ingestion-paths.png",
    "assets/diagrams/index-cost-comparison.png",
    "assets/diagrams/execution-routing.png",
    "assets/diagrams/kernel-result-pipeline.png",
    "assets/diagrams/morsel-aggregation.png",
    "assets/diagrams/morsel-execution.png",
    "assets/diagrams/query-plan-rewrites.png",
    "assets/diagrams/query-plan-transformation.png",
    "assets/pycanopy_logo3.png",
    "assets/spatialbench_sf1.png",
    "assets/spatialbench_sf10.png",
)


def on_files(files, config):
    for path in _DOC_ASSETS:
        files.append(File(path, ".", config["site_dir"], config["use_directory_urls"]))
    return files
