import asyncio
from voice_lab.backend import TaskCoordinator


async def test_cancel_drops_stale_result():
    started = asyncio.Event()

    class Slow:
        async def run(self, task, report):
            started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                return "旧结果"

    outputs = []

    async def report(event):
        pass

    async def deliver(version, value):
        outputs.append(value)

    coordinator = TaskCoordinator(Slow(), report, deliver)
    await coordinator.submit("旧任务")
    await started.wait()
    await coordinator.cancel()
    assert outputs == []


async def test_new_task_replaces_old():
    class Echo:
        async def run(self, task, report):
            await asyncio.sleep(0)
            return task

    outputs = []

    async def report(event):
        pass

    async def deliver(version, value):
        outputs.append(value["result"])

    coordinator = TaskCoordinator(Echo(), report, deliver)
    await coordinator.submit("旧任务")
    await coordinator.submit("新任务")
    await coordinator.task
    assert outputs == ["新任务"]


async def test_error_becomes_result():
    class Failure:
        async def run(self, task, report):
            raise TimeoutError()

    outputs = []

    async def report(event):
        pass

    async def deliver(version, value):
        outputs.append(value)

    coordinator = TaskCoordinator(Failure(), report, deliver)
    await coordinator.submit("任务")
    await coordinator.task
    assert outputs[0]["status"] == "error"


async def test_pending_notice_is_sent_once_per_active_task():
    gate = asyncio.Event()

    class Slow:
        async def run(self, task, report):
            await gate.wait()
            return "真实结果"

    async def report(event):
        pass

    async def deliver(version, value):
        pass

    coordinator = TaskCoordinator(Slow(), report, deliver)
    assert coordinator.pending_context() is None
    await coordinator.submit("查询任务")
    assert "尚未返回" in coordinator.pending_context()
    assert coordinator.pending_context() is None
    gate.set()
    await coordinator.task
    assert coordinator.pending_context() is None
    gate.clear()
    await coordinator.submit("另一个任务")
    assert "尚未返回" in coordinator.pending_context()
    await coordinator.cancel()
    assert coordinator.pending_context() is None
