import docker


class Controller:

    def __init__(self):
        self.client = docker.from_env()

    def apply(self, container_id, cpu, memory):
        try:
            container = self.client.containers.get(container_id)

            cpu_period = 100000
            cpu_quota = int(cpu * cpu_period)

            self.client.api.update_container(
                container.id,
                cpu_period=cpu_period,
                cpu_quota=cpu_quota,
                mem_limit=f"{memory}m",
                memswap_limit=f"{memory}m"
            )

            return {
                "success": True,
                "applied": {
                    "cpu": cpu,
                    "memory": memory
                }
            }

        except Exception as e:
            return {
                "success": False,
                "applied": {},
                "error": str(e)
            }     
                                                                   

                            
          
