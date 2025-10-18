import React, { useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import { gsap } from "gsap";

export default function CTAPrediction() {
  const sectionRef = useRef(null);

  useEffect(() => {
    const grid = sectionRef.current;
    if (!grid) return;

    const spotlightRadius = 260;
    const glowColor = "34, 104, 57"; 

    const spotlight = document.createElement("div");
    spotlight.className = "global-spotlight";
    spotlight.style.cssText = `
      position: fixed;
      width: ${spotlightRadius * 2}px;
      height: ${spotlightRadius * 2}px;
      border-radius: 50%;
      pointer-events: none;
      background: radial-gradient(circle,
        rgba(${glowColor}, 0.14) 0%,
        rgba(${glowColor}, 0.06) 20%,
        transparent 60%);
      z-index: 200;
      opacity: 0;
      transform: translate(-50%, -50%);
      mix-blend-mode: screen;
      transition: opacity 0.12s linear;
    `;
    document.body.appendChild(spotlight);

    const cards = () => grid.querySelectorAll(".card");

    const calc = (mouseX, mouseY, elRect) => {
      const cx = elRect.left + elRect.width / 2;
      const cy = elRect.top + elRect.height / 2;
      const distance = Math.hypot(mouseX - cx, mouseY - cy) - Math.max(elRect.width, elRect.height) / 2;
      return Math.max(0, distance);
    };

    const proximity = spotlightRadius * 0.5;
    const fadeDistance = spotlightRadius * 0.75;

    const onMove = (e) => {
      const rect = grid.getBoundingClientRect();
      const inside =
        e.clientX >= rect.left && e.clientX <= rect.right && e.clientY >= rect.top && e.clientY <= rect.bottom;

      if (!inside) {
        gsap.to(spotlight, { opacity: 0, duration: 0.2, ease: "power2.out" });
        cards().forEach(c => c.style.setProperty("--glow-intensity", "0"));
        return;
      }

      let minDistance = Infinity;
      cards().forEach(card => {
        const r = card.getBoundingClientRect();
        const d = calc(e.clientX, e.clientY, r);
        minDistance = Math.min(minDistance, d);

        const relativeX = ((e.clientX - r.left) / r.width) * 100;
        const relativeY = ((e.clientY - r.top) / r.height) * 100;

        let glow = 0;
        if (d <= proximity) glow = 1;
        else if (d <= fadeDistance) glow = (fadeDistance - d) / (fadeDistance - proximity);

        card.style.setProperty("--glow-x", `${relativeX}%`);
        card.style.setProperty("--glow-y", `${relativeY}%`);
        card.style.setProperty("--glow-intensity", glow.toString());
        card.style.setProperty("--glow-radius", `${spotlightRadius}px`);
      });

      gsap.to(spotlight, { left: e.clientX, top: e.clientY, duration: 0.08, ease: "power2.out" });

      const targetOpacity =
        minDistance <= proximity ? 0.6 : minDistance <= fadeDistance ? ((fadeDistance - minDistance) / (fadeDistance - proximity)) * 0.6 : 0;

      gsap.to(spotlight, { opacity: targetOpacity, duration: 0.12, ease: "power2.out" });
    };

    const onLeave = () => {
      cards().forEach(c => c.style.setProperty("--glow-intensity", "0"));
      gsap.to(spotlight, { opacity: 0, duration: 0.18, ease: "power2.out" });
    };

    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseleave", onLeave);

    return () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseleave", onLeave);
      spotlight.remove();
    };
  }, []);

  return (
    <section ref={sectionRef} className="max-w-7xl mx-auto mt-16 mb-8 px-4">
      <style>
        {`
          .card {
            --glow-x: 50%;
            --glow-y: 50%;
            --glow-intensity: 0;
            --glow-radius: 220px;
            --glow-color: 34, 104, 57;
            position: relative;
            transition: box-shadow .25s ease, transform .2s ease;
          }
          .card::after{
            content: '';
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at var(--glow-x) var(--glow-y),
              rgba(var(--glow-color), calc(var(--glow-intensity) * 0.9)) 0%,
              rgba(var(--glow-color), calc(var(--glow-intensity) * 0.4)) 25%,
              transparent 55%);
            border-radius: inherit;
            pointer-events: none;
            z-index: 0;
            mix-blend-mode: screen;
            opacity: calc(var(--glow-intensity));
            transition: opacity .18s linear;
          }
        `}
      </style>

      <h2 className="text-4xl md:text-5xl font-extrabold text-center mb-4">
        Curious How This AI Works?
      </h2>
      <p className="text-center text-gray-300 max-w-3xl mx-auto mb-8">
        Dive into the code behind the predictions. Explore the XGBoost model, data preprocessing, and machine learning architecture powering these insights.
      </p>

      <div className="card bg-gradient-to-br from-white/6 to-transparent border border-white/6 rounded-3xl p-6 flex flex-col md:flex-row items-center md:items-stretch gap-6">
        <div className="flex-1 p-6" style={{ zIndex: 2 }}>
          <h2 className="text-3xl md:text-4xl font-bold">
            Explore the Source Code
          </h2>
          <p className="text-gray-300 text-sm mt-4 mb-16">
            This prediction system uses XGBoost machine learning algorithm trained on comprehensive EPL historical data. 
            Want to see how it works? Check out the complete source code, model training process, feature engineering, 
            and API implementation on GitHub. Perfect for developers, data scientists, and football analytics enthusiasts!
          </p>
          <a 
            href="https://github.com/kullal/Premier-League-Prediction-Using-XGBoost" 
            target="_blank" 
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2"
          >
            <span className="inline-block py-2 px-8 bg-white/5 backdrop-blur-md border border-white/10 shadow-2xl text-lg font-semibold text-white rounded-full hover:bg-white/15 transition-colors duration-300" style={{ zIndex: 2 }}>
              <svg className="inline-block w-5 h-5 mr-2 -mt-1" fill="currentColor" viewBox="0 0 24 24">
                <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
              </svg>
              View on GitHub
            </span>
          </a>
        </div>

        <div className="hidden md:flex w-1/3 p-6 items-center justify-center" style={{ zIndex: 2 }}>
          <div className="w-full max-w-xs">
            <svg viewBox="0 0 140 80" width="100%" height="100%" aria-hidden="true" role="img">
              <defs>
                <linearGradient id="spark" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0%" stopColor="#34d399" stopOpacity="0.95" />
                  <stop offset="100%" stopColor="#10b981" stopOpacity="0.18" />
                </linearGradient>
                <linearGradient id="area" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0%" stopColor="#10b981" stopOpacity="0.16" />
                  <stop offset="100%" stopColor="#059669" stopOpacity="0.02" />
                </linearGradient>
              </defs>

              <rect x="0" y="0" width="140" height="80" rx="10" fill="url(#area)" />

              <g fill="#e2e8f0" fillOpacity="0.06" transform="translate(18,30)">
                <rect x="0" y="18" width="6" height="18" rx="3" />
                <rect x="20" y="10" width="6" height="26" rx="3" />
                <rect x="40" y="6" width="6" height="30" rx="3" />
                <rect x="60" y="12" width="6" height="24" rx="3" />
                <rect x="80" y="16" width="6" height="20" rx="3" />
              </g>

              <path d="M14 54 C 28 44, 44 38, 58 42 C 72 46, 90 52, 126 34"
                fill="none"
                stroke="url(#spark)"
                strokeWidth="2.2"
                strokeLinecap="round"
                strokeLinejoin="round"
                opacity="0.98"
              />

              <path d="M14 54 C 28 44, 44 38, 58 42 C 72 46, 90 52, 126 34 L126 72 L14 72 Z"
                fill="url(#area)"
              />

              <g fill="#fff" stroke="#34d399" strokeWidth="1">
                <circle cx="14" cy="54" r="2.2" />
                <circle cx="58" cy="42" r="2.2" />
                <circle cx="126" cy="34" r="2.2" />
              </g>
            </svg>
          </div>
        </div>
      </div>
    </section>
  );
}
