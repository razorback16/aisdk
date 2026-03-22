"use client";

import "./App.css";
import { useChat } from "@ai-sdk/react";
import {
	DefaultChatTransport,
	getToolName,
	isToolUIPart,
	type UIMessage,
} from "ai";
import { useEffect, useRef, useState } from "react";

type ToolPart = Extract<UIMessage["parts"][number], { toolCallId: string }>;
type ToolEvent = {
	id: string;
	messageId: string;
	text: string;
};

function formatToolEvent(part: ToolPart) {
	const toolName = getToolName(part);
	switch (part.state) {
		case "input-streaming":
			return part.input == null
				? `Tool: Preparing ${toolName}...`
				: `Tool: Preparing ${toolName}...\n${JSON.stringify(part.input, null, 2)}`;
		case "input-available":
			return `Tool: Executing ${toolName}...\n${JSON.stringify(part.input, null, 2)}`;
		case "output-available":
			return `Tool: Output\n${JSON.stringify(part.output, null, 2)}`;
		case "output-error":
			return `Tool: Error\n${part.errorText}`;
		default:
			return `Tool: ${toolName} [${part.state}]`;
	}
}

export default function App() {
	const { messages, sendMessage, status, error } = useChat({
		transport: new DefaultChatTransport({
			api: "http://localhost:8080/api/chat",
		}),
	});
	const [input, setInput] = useState("what is the weather in new york?");
	const [toolEvents, setToolEvents] = useState<ToolEvent[]>([]);
	// Keep tool updates append-only even though useChat mutates tool parts in place.
	const seenToolEvents = useRef(new Set<string>());

	useEffect(() => {
		if (error) {
			console.error(error);
		}
	}, [error]);

	useEffect(() => {
		const toToolEvent = (messageId: string, part: ToolPart) => {
			const id = `${messageId}:${JSON.stringify(part)}`;
			if (seenToolEvents.current.has(id)) {
				return null;
			}

			seenToolEvents.current.add(id);
			return { id, messageId, text: formatToolEvent(part) };
		};

		const nextEvents = messages.flatMap((message) =>
			message.parts
				.filter(isToolUIPart)
				.map((part) => toToolEvent(message.id, part))
				.filter((event): event is ToolEvent => event !== null),
		);

		if (nextEvents.length > 0) {
			setToolEvents((current) => [...current, ...nextEvents]);
		}
	}, [messages]);

	return (
		<>
			{messages.map((message) => {
				const text = message.parts
					.filter((part) => part.type === "text")
					.map((part) => part.text)
					.join("");

				if (message.role === "user") {
					return <div key={message.id}>User: {text}</div>;
				}

				const events = toolEvents.filter(
					(entry) => entry.messageId === message.id,
				);
				return (
					<div key={message.id}>
						{events.map((entry) => (
							<pre key={entry.id}>{entry.text}</pre>
						))}
						{text ? <div>AI: {text}</div> : null}
					</div>
				);
			})}

			<form
				onSubmit={(e) => {
					e.preventDefault();
					if (input.trim()) {
						sendMessage({ text: input });
						setInput("");
					}
				}}
			>
				<input
					value={input}
					onChange={(e) => setInput(e.target.value)}
					disabled={status !== "ready"}
					placeholder="Say something..."
				/>
				<button type="submit" disabled={status !== "ready"}>
					Submit
				</button>
			</form>
		</>
	);
}
