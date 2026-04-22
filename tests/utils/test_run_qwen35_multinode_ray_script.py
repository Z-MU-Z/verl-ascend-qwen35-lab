# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import subprocess
import textwrap
import tempfile
import unittest
from pathlib import Path

BOOTSTRAP_BASENAME = "bootstrap_remote_qwen35_xpoints_container.sh"
CONTAINER_NAME = "qwen3.5-xpoints"
INSPECT_FORMAT = "{{.Name}}|{{.State.Running}}"
INSPECT_RUNNING = f"/{CONTAINER_NAME}|true"
INSPECT_STOPPED = f"/{CONTAINER_NAME}|false"


def _write_fake_bin(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _read_log(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").strip().splitlines()


class TestRunQwen35MultinodeRayScript(unittest.TestCase):
    def _run_launcher(
        self,
        remote_container_state: str,
        *,
        local_probe_state: str = "running",
        socket_ifname: str | None = None,
        expect_success: bool = True,
    ) -> tuple[int, str, str, list[str]]:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            source_repo = Path(__file__).resolve().parents[2]
            source_launcher = source_repo / "scripts" / "ascend" / "run_qwen35_multinode_ray.sh"
            main_checkout = tmp_path / "XPoints"
            repo = main_checkout / ".worktrees" / "multinode-qwen35-ray-launcher"
            launcher = repo / "scripts" / "ascend" / "run_qwen35_multinode_ray.sh"
            log = tmp_path / "cmd.log"
            if remote_container_state == "running":
                remote_probe_stdout = INSPECT_RUNNING
                remote_probe_stderr = ""
                remote_probe_exit = 0
            elif remote_container_state == "stopped":
                remote_probe_stdout = INSPECT_STOPPED
                remote_probe_stderr = ""
                remote_probe_exit = 0
            elif remote_container_state == "missing":
                remote_probe_stdout = ""
                remote_probe_stderr = f"Error: No such container: {CONTAINER_NAME}"
                remote_probe_exit = 1
            elif remote_container_state == "wrong_identity":
                remote_probe_stdout = "/unexpected-container|true"
                remote_probe_stderr = ""
                remote_probe_exit = 0
            elif remote_container_state == "malformed":
                remote_probe_stdout = "true"
                remote_probe_stderr = ""
                remote_probe_exit = 0
            elif remote_container_state == "probe_failure":
                remote_probe_stdout = ""
                remote_probe_stderr = "ssh: connection failed"
                remote_probe_exit = 255
            elif remote_container_state == "bannered_running":
                remote_probe_stdout = f"Authorized users only. All activities may be monitored and reported.\n{INSPECT_RUNNING}"
                remote_probe_stderr = ""
                remote_probe_exit = 0
            else:
                raise ValueError(f"unsupported remote_container_state: {remote_container_state}")
            if local_probe_state == "running":
                local_probe_stdout = INSPECT_RUNNING
                local_probe_stderr = ""
                local_probe_exit = 0
            elif local_probe_state == "not_running":
                local_probe_stdout = INSPECT_STOPPED
                local_probe_stderr = ""
                local_probe_exit = 0
            elif local_probe_state == "missing":
                local_probe_stdout = ""
                local_probe_stderr = f"Error: No such container: {CONTAINER_NAME}"
                local_probe_exit = 1
            elif local_probe_state == "wrong_identity":
                local_probe_stdout = "/unexpected-container|true"
                local_probe_stderr = ""
                local_probe_exit = 0
            elif local_probe_state == "probe_failure":
                local_probe_stdout = ""
                local_probe_stderr = "Cannot connect to the Docker daemon"
                local_probe_exit = 125
            else:
                raise ValueError(f"unsupported local_probe_state: {local_probe_state}")
            helper = main_checkout / "scripts" / "ascend" / BOOTSTRAP_BASENAME

            launcher.parent.mkdir(parents=True)
            launcher.write_text(source_launcher.read_text(encoding="utf-8"), encoding="utf-8")
            launcher.chmod(0o755)
            helper.parent.mkdir(parents=True)
            helper.write_text(
                textwrap.dedent(
                    f"""\
                    #!/bin/sh
                    echo "bootstrap: REMOTE_HOST=$REMOTE_HOST REMOTE_USER=$REMOTE_USER" >> "{log!s}"
                    """
                ),
                encoding="utf-8",
            )
            helper.chmod(0o755)

            bin_dir = tmp_path / "bin"
            bin_dir.mkdir()
            _write_fake_bin(
                bin_dir / "sudo",
                textwrap.dedent(
                    f"""\
                    #!/bin/sh
                    echo "sudo: $*" >> "{log!s}"
                    if [ "$1" = "-n" ]; then
                      shift
                    fi
                    exec "$@"
                    """
                ),
            )
            _write_fake_bin(
                bin_dir / "docker",
                textwrap.dedent(
                    f"""\
                    #!/bin/sh
                    echo "docker: $*" >> "{log!s}"
                    case "$*" in
                      *"container inspect --format"*"{CONTAINER_NAME}"*)
                        if [ -n "{local_probe_stdout}" ]; then
                          printf '%s\\n' "{local_probe_stdout}"
                        fi
                        if [ -n "{local_probe_stderr}" ]; then
                          printf '%s\\n' "{local_probe_stderr}" >&2
                        fi
                        exit {local_probe_exit}
                        ;;
                    esac
                    exit 0
                    """
                ),
            )
            _write_fake_bin(
                bin_dir / "ray",
                textwrap.dedent(
                    f"""\
                    #!/bin/sh
                    echo "ray: $*" >> "{log!s}"
                    """
                ),
            )
            _write_fake_bin(
                bin_dir / "ssh",
                textwrap.dedent(
                    f"""\
                    #!/bin/sh
                    echo "ssh: $*" >> "{log!s}"
                    case "$*" in
                      *"docker container inspect --format"*"{CONTAINER_NAME}"*)
                        if [ -n "{remote_probe_stdout}" ]; then
                          printf '%s\\n' "{remote_probe_stdout}"
                        fi
                        if [ -n "{remote_probe_stderr}" ]; then
                          printf '%s\\n' "{remote_probe_stderr}" >&2
                        fi
                        exit {remote_probe_exit}
                        ;;
                      *)
                        exit 0
                        ;;
                    esac
                    """
                ),
            )
            _write_fake_bin(
                bin_dir / "python3",
                textwrap.dedent(
                    f"""\
                    #!/bin/sh
                    echo "python3: $*" >> "{log!s}"
                    if [ "$1" = "-" ]; then
                      cat >/dev/null
                    fi
                    if [ "$#" -ge 2 ]; then
                      printf '172.20.117.36\\n'
                    fi
                    """
                ),
            )

            env = {
                **os.environ,
                "PATH": f"{bin_dir}:/usr/bin:/bin",
                "PYTHONNOUSERSITE": "1",
                "LOG_DIR": str(tmp_path / "logs"),
            }
            if socket_ifname is not None:
                env["SOCKET_IFNAME"] = socket_ifname
            remote = "trainer@n1.example"
            r = subprocess.run(
                ["bash", str(launcher), "--remote-ssh", remote],
                env=env,
                cwd=str(repo),
                check=False,
                capture_output=True,
                text=True,
            )
            if expect_success:
                self.assertEqual(
                    r.returncode,
                    0,
                    msg=f"stderr: {r.stderr!r} stdout: {r.stdout!r}",
                )
            lines = _read_log(log)
            return r.returncode, r.stdout, r.stderr, lines

    def _assert_common_launcher_behavior(self, lines: list[str]) -> None:
        ssh_lines = [line for line in lines if line.startswith("ssh: ")]
        docker_lines = [line for line in lines if line.startswith("docker: ")]
        joined = "\n".join(lines)

        self.assertTrue(
            any(
                "container inspect --format" in line and CONTAINER_NAME in line
                for line in docker_lines
            ),
            msg="expected local head container check via docker inspect",
        )
        self.assertTrue(
            "ray stop --force" in joined,
            msg="expected ray stop before head/worker startup",
        )
        self.assertGreaterEqual(len(ssh_lines), 1)
        self.assertTrue(
            any(
                "docker container inspect --format" in line and CONTAINER_NAME in line
                for line in ssh_lines
            )
        )
        self.assertTrue(
            any(
                line.startswith("ssh: ") and "sudo -n docker" in line
                for line in ssh_lines
            ),
            msg="expected remote docker access via sudo -n docker",
        )
        self.assertFalse(
            any("StrictHostKeyChecking=no" in line for line in ssh_lines),
            msg="launcher should not disable strict host key checking",
        )
        self.assertTrue(
            "docker: exec qwen3.5-xpoints bash -lc " in joined and "ray start --head" in joined,
            msg="expected local ray head startup in container",
        )
        self.assertTrue(
            "ssh: -o BatchMode=yes trainer@n1.example sudo -n docker exec qwen3.5-xpoints bash -lc" in joined
            and "ray start --address=" in joined,
            msg="expected remote ray worker startup in container",
        )
        self.assertTrue(
            '--resources=\'{"NPU": 8}\'' in joined
            or '--resources="{\\"NPU\\": 8}"' in joined
            or '--resources=\\\'{"NPU": 8}\\\'' in joined,
            msg="expected explicit NPU resource registration for Ray worker",
        )
        self.assertTrue(
            "run_qwen3_5_4b_vllm_fsdp_npu_container_clean.sh" in joined,
            msg="expected local container training launch",
        )
        self.assertTrue(any("trainer.nnodes='2'" in line for line in lines))
        self.assertTrue(any("trainer.n_gpus_per_node='8'" in line for line in lines))
        self.assertTrue(any("python3: -" in line for line in lines))

    @staticmethod
    def _first_index_startswith(lines: list[str], prefix: str) -> int:
        for i, line in enumerate(lines):
            if line.startswith(prefix):
                return i
        return -1

    @staticmethod
    def _find_command_block_start(lines: list[str], prefix: str, needle: str) -> int:
        for i, line in enumerate(lines):
            if line.startswith(prefix):
                window = "\n".join(lines[i : i + 12])
                if needle in window:
                    return i
        return -1

    def test_bootstrap_runs_when_remote_container_is_missing(self) -> None:
        _, _, _, lines = self._run_launcher(remote_container_state="missing")
        self._assert_common_launcher_behavior(lines)

        ssh_lines = [line for line in lines if line.startswith("ssh: ")]
        bootstrap_lines = [line for line in lines if line.startswith("bootstrap: ")]
        self.assertEqual(len(ssh_lines), 4)
        self.assertEqual(len(bootstrap_lines), 1)
        self.assertIn("REMOTE_HOST=n1.example", bootstrap_lines[0])
        self.assertIn("REMOTE_USER=trainer", bootstrap_lines[0])
        self.assertTrue(any(line.startswith("sudo: -n docker ") for line in lines))

    def test_bootstrap_is_skipped_when_remote_container_is_present(self) -> None:
        _, _, _, lines = self._run_launcher(remote_container_state="running")
        self._assert_common_launcher_behavior(lines)

        ssh_lines = [line for line in lines if line.startswith("ssh: ")]
        bootstrap_lines = [line for line in lines if line.startswith("bootstrap: ")]
        self.assertEqual(len(ssh_lines), 4)
        self.assertEqual(bootstrap_lines, [])

    def test_remote_probe_ignores_login_banner(self) -> None:
        _, _, _, lines = self._run_launcher(remote_container_state="bannered_running")
        self._assert_common_launcher_behavior(lines)

    def test_exports_socket_ifname_family_when_provided(self) -> None:
        _, _, _, lines = self._run_launcher(remote_container_state="running", socket_ifname="enp189s0f0")
        joined = "\n".join(lines)
        self.assertIn("export SOCKET_IFNAME='enp189s0f0'", joined)
        self.assertIn('export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${SOCKET_IFNAME}}"', joined)
        self.assertIn('export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-${SOCKET_IFNAME}}"', joined)
        self.assertIn('export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${SOCKET_IFNAME}}"', joined)

    def test_exits_before_ray_when_local_head_container_is_not_running(self) -> None:
        code, out, err, lines = self._run_launcher(
            remote_container_state="running",
            local_probe_state="not_running",
            expect_success=False,
        )
        self.assertNotEqual(code, 0, msg=err)
        self.assertTrue(
            CONTAINER_NAME in err or CONTAINER_NAME in out,
            msg=f"stderr={err!r} stdout={out!r}",
        )
        self.assertEqual([], [line for line in lines if line.startswith("ray: ")])
        self.assertEqual([], [line for line in lines if line.startswith("ssh: ")])

    def test_exits_before_remote_work_when_local_docker_probe_fails(self) -> None:
        code, out, err, lines = self._run_launcher(
            remote_container_state="running",
            local_probe_state="probe_failure",
            expect_success=False,
        )
        self.assertNotEqual(code, 0, msg=err)
        self.assertTrue(
            "local docker probe failed" in err or "local docker probe failed" in out,
            msg=f"stderr={err!r} stdout={out!r}",
        )
        self.assertEqual([], [line for line in lines if line.startswith("ssh: ")])
        self.assertEqual([], [line for line in lines if line.startswith("bootstrap: ")])
        self.assertFalse(any("ray start" in line for line in lines))

    def test_treats_exact_local_missing_container_as_not_running(self) -> None:
        code, out, err, lines = self._run_launcher(
            remote_container_state="running",
            local_probe_state="missing",
            expect_success=False,
        )
        self.assertNotEqual(code, 0, msg=err)
        self.assertTrue(
            f"local head container '{CONTAINER_NAME}' is not running" in err
            or f"local head container '{CONTAINER_NAME}' is not running" in out,
            msg=f"stderr={err!r} stdout={out!r}",
        )
        self.assertEqual([], [line for line in lines if line.startswith("ssh: ")])
        self.assertFalse(any("ray start" in line for line in lines))

    def test_exits_before_remote_work_when_local_probe_reports_wrong_container(self) -> None:
        code, out, err, lines = self._run_launcher(
            remote_container_state="running",
            local_probe_state="wrong_identity",
            expect_success=False,
        )
        self.assertNotEqual(code, 0, msg=err)
        self.assertTrue(
            "unexpected local docker probe result" in err or "unexpected local docker probe result" in out,
            msg=f"stderr={err!r} stdout={out!r}",
        )
        self.assertEqual([], [line for line in lines if line.startswith("ssh: ")])
        self.assertEqual([], [line for line in lines if line.startswith("bootstrap: ")])
        self.assertFalse(any("ray start" in line for line in lines))

    def test_bootstrap_runs_when_remote_container_exists_but_is_stopped(self) -> None:
        _, _, _, lines = self._run_launcher(remote_container_state="stopped")
        self._assert_common_launcher_behavior(lines)

        ssh_lines = [line for line in lines if line.startswith("ssh: ")]
        bootstrap_lines = [line for line in lines if line.startswith("bootstrap: ")]
        self.assertEqual(len(ssh_lines), 4)
        self.assertEqual(len(bootstrap_lines), 1)

    def test_remote_probe_failure_exits_before_bootstrap_and_ray(self) -> None:
        code, out, err, lines = self._run_launcher(
            remote_container_state="probe_failure",
            expect_success=False,
        )
        self.assertNotEqual(code, 0, msg=err)
        self.assertTrue(
            "remote container probe failed" in err or "remote container probe failed" in out,
            msg=f"stderr={err!r} stdout={out!r}",
        )
        self.assertEqual(1, len([line for line in lines if line.startswith("ssh: ")]))
        self.assertEqual([], [line for line in lines if line.startswith("bootstrap: ")])
        self.assertFalse(any("ray start" in line for line in lines))

    def test_remote_probe_wrong_identity_exits_before_bootstrap_and_ray(self) -> None:
        code, out, err, lines = self._run_launcher(
            remote_container_state="wrong_identity",
            expect_success=False,
        )
        self.assertNotEqual(code, 0, msg=err)
        self.assertTrue(
            "unexpected remote probe result" in err or "unexpected remote probe result" in out,
            msg=f"stderr={err!r} stdout={out!r}",
        )
        self.assertEqual(1, len([line for line in lines if line.startswith("ssh: ")]))
        self.assertEqual([], [line for line in lines if line.startswith("bootstrap: ")])
        self.assertFalse(any("ray start" in line for line in lines))

    def test_remote_probe_malformed_output_exits_before_bootstrap_and_ray(self) -> None:
        code, out, err, lines = self._run_launcher(
            remote_container_state="malformed",
            expect_success=False,
        )
        self.assertNotEqual(code, 0, msg=err)
        self.assertTrue(
            "unexpected remote probe result" in err or "unexpected remote probe result" in out,
            msg=f"stderr={err!r} stdout={out!r}",
        )
        self.assertEqual(1, len([line for line in lines if line.startswith("ssh: ")]))
        self.assertEqual([], [line for line in lines if line.startswith("bootstrap: ")])
        self.assertFalse(any("ray start" in line for line in lines))

    def test_remote_bootstrap_prerequisite_orchestration_order(self) -> None:
        """
        Local docker must be validated before the remote worker check, and remote
        bootstrap (if any) must run before Ray and training stages.
        """
        _, _, _, lines = self._run_launcher(remote_container_state="missing")
        joined = "\n".join(lines)
        i_docker = joined.find("docker: container inspect --format")
        i_ssh = joined.find("ssh: -o BatchMode=yes")
        i_bootstrap = joined.find("bootstrap:")
        i_python = joined.find("python3: -")
        i_local_head = joined.find("ray start --head")
        i_remote_worker = joined.find("ray start --address=")
        i_train = joined.find("run_qwen3_5_4b_vllm_fsdp_npu_container_clean.sh' trainer.nnodes='2'")

        self.assertGreaterEqual(i_docker, 0)
        self.assertGreater(i_ssh, i_docker)
        self.assertGreater(i_bootstrap, i_ssh)
        self.assertGreater(i_python, i_bootstrap)
        self.assertGreater(i_local_head, i_python)
        self.assertGreater(i_remote_worker, i_local_head)
        self.assertGreater(i_train, i_remote_worker)


if __name__ == "__main__":
    unittest.main()
