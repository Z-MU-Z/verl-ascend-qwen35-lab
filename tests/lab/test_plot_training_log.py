import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PLOT_SCRIPT = ROOT / "scripts/ascend/plot_training_log.py"


def _run_plot(log_path: Path, output_path: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(PLOT_SCRIPT),
            "--log",
            str(log_path),
            "--output",
            str(output_path),
            *extra_args,
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_plot_script_extracts_step_metrics_and_writes_svg_and_csv(tmp_path: Path) -> None:
    log_path = tmp_path / "train.log"
    output_path = tmp_path / "curve.svg"
    csv_path = tmp_path / "curve.csv"
    log_path.write_text(
        "\n".join(
            [
                "noise before metrics",
                "step:1 - training/global_step:1 - critic/rewards/mean:0.0 - val-core/hiyouga/geometry3k/acc/mean@1:0.02",
                "step:2 - training/global_step:2 - critic/rewards/mean:0.1 - val-core/hiyouga/geometry3k/acc/mean@1:0.05",
                "step:3 - training/global_step:3 - critic/rewards/mean:0.2 - val-core/hiyouga/geometry3k/acc/mean@1:0.08",
            ]
        )
        + "\n"
    )

    result = _run_plot(log_path, output_path, "--csv", str(csv_path))

    assert result.returncode == 0, result.stderr
    assert "Parsed 3 step line(s)." in result.stdout
    assert "val-core/hiyouga/geometry3k/acc/mean@1" in result.stdout
    assert "critic/rewards/mean" in result.stdout
    assert output_path.exists()
    assert "<svg" in output_path.read_text()
    assert csv_path.exists()
    csv_text = csv_path.read_text()
    assert "step,training/global_step,val-core/hiyouga/geometry3k/acc/mean@1,critic/rewards/mean" in csv_text
    assert "1.0,1.0,0.02,0.0" in csv_text
    assert "3.0,3.0,0.08,0.2" in csv_text


def test_plot_script_supports_explicit_metric_subset(tmp_path: Path) -> None:
    log_path = tmp_path / "train.log"
    output_path = tmp_path / "curve.svg"
    log_path.write_text(
        "\n".join(
            [
                "step:1 - training/global_step:1 - critic/rewards/mean:0.0 - actor/entropy:0.2",
                "step:2 - training/global_step:2 - critic/rewards/mean:0.3 - actor/entropy:0.4",
            ]
        )
        + "\n"
    )

    result = _run_plot(log_path, output_path, "--metric", "actor/entropy")

    assert result.returncode == 0, result.stderr
    svg_text = output_path.read_text()
    assert "actor/entropy" in svg_text
    assert "critic/rewards/mean" not in result.stdout


def test_plot_script_fails_when_no_step_metrics_are_found(tmp_path: Path) -> None:
    log_path = tmp_path / "train.log"
    output_path = tmp_path / "curve.svg"
    log_path.write_text("just warnings\nno metrics here\n")

    result = _run_plot(log_path, output_path)

    assert result.returncode != 0
    assert "No step metrics found" in result.stderr
