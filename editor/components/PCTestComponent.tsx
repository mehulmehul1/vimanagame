import React, { useEffect, useRef } from 'react';
import { Panel, Button, NumericInput, Label } from '@playcanvas/pcui';
import '@playcanvas/pcui/styles.css';

/**
 * PCTestComponent - Tests PCUI rendering
 *
 * This component verifies that PCUI is properly installed and configured.
 * It tests basic PCUI components: Panel, Button, NumericInput
 */
const PCTestComponent: React.FC = () => {
    const containerRef = useRef<HTMLDivElement>(null);
    const numericInputRef = useRef<any>(null);

    useEffect(() => {
        if (!containerRef.current) return;

        console.log('PCUI Test Component mounted');
        console.log('Testing PCUI components...');

        // Create PCUI components
        const panel = new Panel({
            headerText: 'PCUI Test Panel',
        });

        const button = new Button({
            text: 'Click Me',
            onClick: () => {
                console.log('PCUI Button clicked!');
                alert('PCUI is working correctly!');
            }
        });

        const numericInput = new NumericInput({
            value: 42,
            min: 0,
            max: 100,
            placeholder: 'Enter a number',
            onChange: (value: number) => {
                console.log('Numeric input changed:', value);
            }
        });
        numericInputRef.current = numericInput;

        const label = new Label({
            text: 'Numeric Input Test:'
        });

        // Append components to panel
        panel.append(label);
        panel.append(numericInput);
        panel.append(button);

        // Append panel to DOM
        containerRef.current.appendChild(panel.dom);

        // Cleanup
        return () => {
            console.log('PCUI Test Component unmounting');
            if (containerRef.current && panel.dom.parentElement === containerRef.current) {
                containerRef.current.removeChild(panel.dom);
            }
            panel.destroy();
        };
    }, []);

    return (
        <div
            ref={containerRef}
            style={{
                padding: '20px',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                minHeight: '200px'
            }}
        >
            <p style={{ color: '#888', fontSize: '12px' }}>Loading PCUI test...</p>
        </div>
    );
};

export default PCTestComponent;
