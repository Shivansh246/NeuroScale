import unittest
from unittest.mock import MagicMock, patch
from controller.controller import Controller
from docker.errors import NotFound, APIError, DockerException


class TestController(unittest.TestCase):

    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_container = MagicMock()
        self.mock_container.id = "test_container_12345"
        self.mock_client.containers.get.return_value = self.mock_container
        self.controller = Controller(client=self.mock_client)

    def test_successful_resource_update(self):
        result = self.controller.apply("test_container_12345", cpu=1.5, memory=512)

        self.assertTrue(result["success"])
        self.assertEqual(result["container_id"], "test_container_12345")
        self.assertIsNone(result["error"])
        self.assertEqual(result["applied"]["cpu"], 1.5)
        self.assertEqual(result["applied"]["memory"], 512)
        self.assertEqual(result["applied"]["cpu_period"], 100000)
        self.assertEqual(result["applied"]["cpu_quota"], 150000)
        self.assertEqual(result["applied"]["mem_limit"], "512m")

        self.mock_client.api.update_container.assert_called_once_with(
            "test_container_12345",
            cpu_period=100000,
            cpu_quota=150000,
            mem_limit="512m",
            memswap_limit="512m",
        )

    def test_validation_invalid_container_id(self):
        result = self.controller.apply("", cpu=1.0, memory=256)
        self.assertFalse(result["success"])
        self.assertIn("container_id", result["error"])

        result = self.controller.apply(None, cpu=1.0, memory=256)
        self.assertFalse(result["success"])
        self.assertIn("container_id", result["error"])

    def test_validation_cpu_bounds(self):
        # Below min
        result = self.controller.apply("c1", cpu=0.01, memory=256)
        self.assertFalse(result["success"])
        self.assertIn("out of bounds", result["error"])

        # Above max
        result = self.controller.apply("c1", cpu=32.0, memory=256)
        self.assertFalse(result["success"])
        self.assertIn("out of bounds", result["error"])

        # Non-numeric
        result = self.controller.apply("c1", cpu="fast", memory=256)
        self.assertFalse(result["success"])
        self.assertIn("numeric", result["error"])

        # Boolean
        result = self.controller.apply("c1", cpu=True, memory=256)
        self.assertFalse(result["success"])
        self.assertIn("numeric", result["error"])

    def test_validation_memory_bounds(self):
        # Below min
        result = self.controller.apply("c1", cpu=1.0, memory=10)
        self.assertFalse(result["success"])
        self.assertIn("out of bounds", result["error"])

        # Above max
        result = self.controller.apply("c1", cpu=1.0, memory=100000)
        self.assertFalse(result["success"])
        self.assertIn("out of bounds", result["error"])

        # Non-numeric
        result = self.controller.apply("c1", cpu=1.0, memory="1g")
        self.assertFalse(result["success"])
        self.assertIn("numeric", result["error"])

    def test_custom_bounds_configuration(self):
        custom_ctrl = Controller(
            client=self.mock_client,
            min_cpu=0.5,
            max_cpu=4.0,
            min_memory_mb=128,
            max_memory_mb=1024,
        )
        # 0.2 is below custom min 0.5
        res = custom_ctrl.apply("c1", cpu=0.2, memory=256)
        self.assertFalse(res["success"])
        self.assertIn("out of bounds", res["error"])

        # 0.5 is valid
        res = custom_ctrl.apply("c1", cpu=0.5, memory=256)
        self.assertTrue(res["success"])

    def test_container_not_found_handling(self):
        self.mock_client.containers.get.side_effect = NotFound("Container not found")

        result = self.controller.apply("missing_container", cpu=1.0, memory=256)
        self.assertFalse(result["success"])
        self.assertEqual(result["container_id"], "missing_container")
        self.assertIn("Container not found", result["error"])

    def test_docker_api_error_handling(self):
        self.mock_client.api.update_container.side_effect = APIError("Server error 500")

        result = self.controller.apply("test_container", cpu=1.0, memory=256)
        self.assertFalse(result["success"])
        self.assertIn("Docker API error", result["error"])

    def test_docker_daemon_connection_error(self):
        self.mock_client.containers.get.side_effect = DockerException("Connection refused")

        result = self.controller.apply("test_container", cpu=1.0, memory=256)
        self.assertFalse(result["success"])
        self.assertIn("Docker daemon error", result["error"])

    def test_swap_unsupported_fallback(self):
        # First call with memswap_limit raises swap-related APIError, second call succeeds
        def update_side_effect(*args, **kwargs):
            if "memswap_limit" in kwargs:
                raise APIError("Your kernel does not support swap limit capabilities")
            return {}

        self.mock_client.api.update_container.side_effect = update_side_effect

        result = self.controller.apply("test_container_12345", cpu=1.0, memory=256)
        self.assertTrue(result["success"])
        self.assertEqual(self.mock_client.api.update_container.call_count, 2)


if __name__ == "__main__":
    unittest.main()
