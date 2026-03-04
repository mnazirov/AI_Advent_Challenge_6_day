from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _run_node_case(case_js: str) -> None:
    if not shutil.which("node"):
        pytest.skip("node is not installed")

    harness = textwrap.dedent(
        r"""
        import assert from 'node:assert/strict';
        import fs from 'node:fs';
        import vm from 'node:vm';

        const html = fs.readFileSync('templates/index.html', 'utf8');
        const match = html.match(/<script>([\s\S]*?)<\/script>\s*<\/body>/i) || html.match(/<script>([\s\S]*?)<\/script>/i);
        assert.ok(match, 'inline script not found in templates/index.html');
        const script = match[1];

        class MockClassList {
          add() {}
          remove() {}
          toggle() { return false; }
          contains() { return false; }
        }

        class MockElement {
          constructor(id = '') {
            this.id = id;
            this.children = [];
            this.innerHTML = '';
            this.textContent = '';
            this.value = '';
            this.disabled = false;
            this.options = [];
            this.style = {};
            this.classList = new MockClassList();
            this.scrollHeight = 0;
            this.scrollTop = 0;
          }
          addEventListener() {}
          removeEventListener() {}
          appendChild(node) { this.children.push(node); return node; }
          remove() {}
          focus() {}
          querySelectorAll() { return []; }
          getContext() { return { clearRect() {}, fillRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {} }; }
        }

        const store = new Map();
        const actionButtons = [new MockElement('ab-1'), new MockElement('ab-2')];
        const ensureEl = (id) => {
          if (!store.has(id)) store.set(id, new MockElement(id));
          return store.get(id);
        };

        const documentMock = {
          body: new MockElement('body'),
          documentElement: new MockElement('html'),
          getElementById(id) { return ensureEl(id); },
          querySelectorAll(sel) { return sel === '.task-action-btn' ? actionButtons : []; },
          querySelector() { return null; },
          createElement(tag) { return new MockElement(tag); },
        };
        documentMock.documentElement.style = { setProperty() {} };

        globalThis.window = globalThis;
        globalThis.document = documentMock;
        globalThis.navigator = { userAgent: 'node' };
        globalThis.location = { reload() { globalThis.__reloaded = true; } };
        globalThis.localStorage = { getItem() { return null; }, setItem() {}, removeItem() {} };
        globalThis.requestAnimationFrame = (cb) => setTimeout(cb, 0);
        globalThis.cancelAnimationFrame = (id) => clearTimeout(id);
        globalThis.fetch = async () => ({ ok: true, headers: { get: () => 'application/json' }, json: async () => ({ reply: 'ok' }) });
        window.addEventListener = () => {};
        window.removeEventListener = () => {};

        vm.runInThisContext(script, { filename: 'templates/index.html::<script>' });
        """
    )

    proc = subprocess.run(
        ["node", "--input-type=module", "-e", harness + "\n" + case_js],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        "Node scenario failed\n"
        f"STDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}"
    )


def test_network_error_does_not_block_input() -> None:
    case = textwrap.dedent(
        r"""
        document.getElementById('userInput').value = 'test message';
        globalThis.fetch = async () => { throw new TypeError('Failed to fetch'); };

        await vm.runInThisContext('sendMsg()');

        assert.equal(vm.runInThisContext('isBusy'), false, 'isBusy must be released in finally');
        assert.equal(document.getElementById('userInput').disabled, false, 'textarea must stay enabled after error');
        assert.equal(document.getElementById('sendBtn').disabled, false, 'send button must be re-enabled after error');
        """
    )
    _run_node_case(case)


def test_action_click_guard_blocks_double_send_when_busy() -> None:
    case = textwrap.dedent(
        r"""
        let fetchCalls = 0;
        globalThis.fetch = async () => {
          fetchCalls += 1;
          return {
            ok: true,
            headers: { get: () => 'application/json' },
            json: async () => ({ reply: 'ok' }),
          };
        };

        vm.runInThisContext('isBusy = true');
        await vm.runInThisContext('handleActionClick({ kind: "message", message: "ping" })');
        assert.equal(fetchCalls, 0, 'busy guard must skip action click');

        vm.runInThisContext('isBusy = false');
        await vm.runInThisContext('handleActionClick({ kind: "message", message: "ping" })');
        assert.equal(fetchCalls, 1, 'second click after release must send exactly one request');
        """
    )
    _run_node_case(case)
