import React, { useState, useEffect } from 'react';
import './CriteriaBuilder.css';

/**
 * Criteria Builder UI Component
 *
 * Features:
 * - State selector dropdown (GAME_STATES enum)
 * - Flag checkboxes (array of strings)
 * - Visual AND/OR builder
 * - Custom criteria expression editor (textarea for advanced)
 * - Preview which objects match criteria
 */

// Available game states (matching GAME_STATES enum from gameData.js)
const GAME_STATES = {
    LOADING: 0,
    INTRO: 1,
    CHAPTER_1: 2,
    CHAPTER_2: 3,
    CHAPTER_3: 4,
    CHAPTER_4: 5,
    ENDING: 6,
    POST_DRIVE_BY: 7,
    ENTERING_OFFICE: 8,
    ANSWERED_PHONE: 9,
    DIALOG_COMPLETE: 10,
    GAME_OVER: 11
};

// Comparison operators
const OPERATORS = {
    $eq: 'Equals',
    $ne: 'Not Equals',
    $gt: 'Greater Than',
    $gte: 'Greater Than or Equal',
    $lt: 'Less Than',
    $lte: 'Less Than or Equal',
    $in: 'In Array',
    $nin: 'Not In Array'
};

interface CriteriaBuilderProps {
    criteria?: Record<string, any>;
    onCriteriaChange?: (criteria: Record<string, any>) => void;
}

interface CriteriaCondition {
    id: string;
    field: string;
    operator: keyof typeof OPERATORS;
    value: string | number;
    logicalOperator?: 'AND' | 'OR';
}

const CriteriaBuilder: React.FC<CriteriaBuilderProps> = ({ criteria = {}, onCriteriaChange }) => {
    const [conditions, setConditions] = useState<CriteriaCondition[]>([]);
    const [advancedMode, setAdvancedMode] = useState(false);
    const [advancedCriteria, setAdvancedCriteria] = useState('');

    // Initialize conditions from props
    useEffect(() => {
        if (criteria && Object.keys(criteria).length > 0) {
            // Check if criteria is simple or complex
            const criteriaKeys = Object.keys(criteria);
            const hasOperators = criteriaKeys.some(key => typeof criteria[key] === 'object');

            if (hasOperators) {
                // Complex criteria with operators
                const newConditions: CriteriaCondition[] = [];

                criteriaKeys.forEach((field, index) => {
                    const fieldCriteria = criteria[field];

                    if (typeof fieldCriteria === 'object') {
                        Object.keys(fieldCriteria).forEach(operator => {
                            newConditions.push({
                                id: `cond-${Date.now()}-${index}`,
                                field,
                                operator: operator as keyof typeof OPERATORS,
                                value: fieldCriteria[operator],
                                logicalOperator: index === 0 ? undefined : 'AND'
                            });
                        });
                    } else {
                        newConditions.push({
                            id: `cond-${Date.now()}-${index}`,
                            field,
                            operator: '$eq',
                            value: fieldCriteria,
                            logicalOperator: index === 0 ? undefined : 'AND'
                        });
                    }
                });

                setConditions(newConditions);
                setAdvancedCriteria(JSON.stringify(criteria, null, 2));
            } else {
                // Simple criteria
                const newConditions: CriteriaCondition[] = criteriaKeys.map((field, index) => ({
                    id: `cond-${Date.now()}-${index}`,
                    field,
                    operator: '$eq',
                    value: criteria[field],
                    logicalOperator: index === 0 ? undefined : 'AND'
                }));

                setConditions(newConditions);
                setAdvancedCriteria(JSON.stringify(criteria, null, 2));
            }
        }
    }, [criteria]);

    /**
     * Add a new condition
     */
    const addCondition = () => {
        const newCondition: CriteriaCondition = {
            id: `cond-${Date.now()}`,
            field: 'currentState',
            operator: '$eq',
            value: GAME_STATES.LOADING,
            logicalOperator: conditions.length > 0 ? 'AND' : undefined
        };

        setConditions([...conditions, newCondition]);
        notifyParent();
    };

    /**
     * Remove a condition
     */
    const removeCondition = (id: string) => {
        const newConditions = conditions.filter(c => c.id !== id);
        setConditions(newConditions);
        notifyParent();
    };

    /**
     * Update a condition
     */
    const updateCondition = (id: string, updates: Partial<CriteriaCondition>) => {
        const newConditions = conditions.map(c => {
            if (c.id === id) {
                return { ...c, ...updates };
            }
            return c;
        });
        setConditions(newConditions);
        notifyParent();
    };

    /**
     * Build criteria object from conditions
     */
    const buildCriteria = (): Record<string, any> => {
        const result: Record<string, any> = {};

        conditions.forEach(cond => {
            if (cond.logicalOperator === 'OR') {
                // Handle OR logic (would need more complex structure)
                // For now, just use AND logic
            }

            if (cond.operator === '$eq') {
                result[cond.field] = cond.value;
            } else {
                if (!result[cond.field]) {
                    result[cond.field] = {};
                }
                result[cond.field][cond.operator] = cond.value;
            }
        });

        return result;
    };

    /**
     * Notify parent of criteria change
     */
    const notifyParent = () => {
        if (onCriteriaChange) {
            setTimeout(() => {
                const criteriaObj = buildCriteria();
                onCriteriaChange(criteriaObj);
            }, 0);
        }
    };

    /**
     * Handle advanced criteria change
     */
    const handleAdvancedCriteriaChange = (value: string) => {
        setAdvancedCriteria(value);

        try {
            const parsed = JSON.parse(value);
            if (onCriteriaChange) {
                onCriteriaChange(parsed);
            }
        } catch (error) {
            // Invalid JSON, don't update
        }
    };

    return (
        <div className="criteria-builder">
            <div className="criteria-header">
                <div className="criteria-title">Visibility Criteria</div>
                <button
                    className={`mode-toggle ${advancedMode ? 'advanced' : ''}`}
                    onClick={() => setAdvancedMode(!advancedMode)}
                >
                    {advancedMode ? 'Simple Builder' : 'Advanced Editor'}
                </button>
            </div>

            {!advancedMode ? (
                <div className="criteria-simple">
                    {conditions.length === 0 ? (
                        <div className="criteria-empty">
                            <p>No criteria set. Object will always be visible.</p>
                            <button className="add-condition-button" onClick={addCondition}>
                                + Add Condition
                            </button>
                        </div>
                    ) : (
                        <div className="conditions-list">
                            {conditions.map((cond, index) => (
                                <div key={cond.id} className="condition-row">
                                    {index > 0 && (
                                        <div className="logical-operator">
                                            {cond.logicalOperator}
                                        </div>
                                    )}

                                    <div className="condition-content">
                                        <select
                                            value={cond.field}
                                            onChange={(e) => updateCondition(cond.id, { field: e.target.value })}
                                            className="condition-field"
                                        >
                                            <option value="currentState">Game State</option>
                                            <option value="performanceProfile">Performance Profile</option>
                                            <option value="customFlag">Custom Flag</option>
                                        </select>

                                        <select
                                            value={cond.operator}
                                            onChange={(e) => updateCondition(cond.id, { operator: e.target.value as keyof typeof OPERATORS })}
                                            className="condition-operator"
                                        >
                                            {Object.entries(OPERATORS).map(([key, label]) => (
                                                <option key={key} value={key}>{label}</option>
                                            ))}
                                        </select>

                                        {cond.field === 'currentState' ? (
                                            <select
                                                value={cond.value as number}
                                                onChange={(e) => updateCondition(cond.id, { value: parseInt(e.target.value) })}
                                                className="condition-value"
                                            >
                                                {Object.entries(GAME_STATES).map(([key, value]) => (
                                                    <option key={key} value={value as number}>{key}</option>
                                                ))}
                                            </select>
                                        ) : cond.field === 'performanceProfile' ? (
                                            <select
                                                value={cond.value as string}
                                                onChange={(e) => updateCondition(cond.id, { value: e.target.value })}
                                                className="condition-value"
                                            >
                                                <option value="max">Max Quality</option>
                                                <option value="laptop">Laptop</option>
                                                <option value="desktop">Desktop</option>
                                                <option value="mobile">Mobile</option>
                                            </select>
                                        ) : (
                                            <input
                                                type="text"
                                                value={cond.value as string}
                                                onChange={(e) => updateCondition(cond.id, { value: e.target.value })}
                                                className="condition-value-input"
                                                placeholder="Value"
                                            />
                                        )}

                                        <button
                                            className="remove-condition-button"
                                            onClick={() => removeCondition(cond.id)}
                                            title="Remove condition"
                                        >
                                            ✕
                                        </button>
                                    </div>
                                </div>
                            ))}

                            <button className="add-condition-button" onClick={addCondition}>
                                + Add Condition
                            </button>
                        </div>
                    )}

                    <div className="criteria-preview">
                        <div className="preview-label">Preview:</div>
                        <pre className="preview-code">{JSON.stringify(buildCriteria(), null, 2)}</pre>
                    </div>
                </div>
            ) : (
                <div className="criteria-advanced">
                    <textarea
                        value={advancedCriteria}
                        onChange={(e) => handleAdvancedCriteriaChange(e.target.value)}
                        className="advanced-editor"
                        placeholder='{"currentState": { "$gte": 0, "$lt": 5 }}'
                        spellCheck={false}
                    />
                    <div className="advanced-hint">
                        Enter criteria as JSON. Use $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin operators.
                    </div>
                </div>
            )}
        </div>
    );
};

export default CriteriaBuilder;
