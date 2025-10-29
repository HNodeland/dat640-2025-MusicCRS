import React from "react";

type Props = {
  onClick: () => void;
  disabled?: boolean;
};

export default function RecommendButton({ onClick, disabled }: Props) {
  return (
    <button
      type="button"
      className="btn btn-outline-primary ms-2 d-inline-flex align-items-center"
      onClick={onClick}
      disabled={disabled}
      aria-label="Get 3 song recommendations"
      title="Get 3 song recommendations"
    >
      {/* small inline icon so you don’t need any extra libraries */}
      <svg
        aria-hidden="true"
        width="16"
        height="16"
        viewBox="0 0 16 16"
        className="me-2"
        focusable="false"
      >
        <path d="M8 1l1.8 3.8L14 6.2l-3 2.6.9 4.2L8 10.9 4.1 13l.9-4.2-3-2.6 4.2-.4L8 1z" fill="currentColor" />
      </svg>
      Recommend
    </button>
  );
}
